"""
Precompute a pool of patches to disk so training is GPU-bound.

The training bottleneck is per-patch BM4D + cloud reads on the CPU, which
leaves the GPU idle -- both for the training pool and for the validation set
that init_datasets otherwise builds live at startup. This script does that work
once, offline, writing the expensive count-space intermediates -- (raw with the
per-brain offset subtracted, clipped BM4D teacher, foreground mask) -- to
memory-mapped arrays. Training then reads a cached patch and applies only the
cheap transform + target construction (see CachedPatchDataset /
CachedValidateDataset), so no cloud access or BM4D happens at startup.

One script builds both caches; ``--split`` selects which:

    python scripts/precompute.py --split train   # GPU-bound training pool
    python scripts/precompute.py --split val      # fixed validation set

Both splits draw voxels with the TrainDataset's foreground-biased sampler and
build the foreground mask from the segmentation labels (used as-is unless
segmentation_dilate > 0) unioned with the traced skeleton (dilated to a neurite
radius), so the training target and the validation metric agree on what counts
as neurite signal -- bright non-neuronal structures (noise,
off-target label) are left for the BM4D teacher to denoise rather than
preserved, while neurites the segmentation misses are still protected by the
skeleton. The train split builds the mask inside TrainDataset; the val split
builds the annotation mask from the TrainDataset and hands it to the
ValidateDataset. The splits otherwise differ only in the outputs each records.

A distinct RNG stream per split means the two caches never sample the same
(brain, voxel) for a given task index when built with the same base seed.

Outputs, under cache_dir (identical layout for both splits, so the val cache
loads with CachedValidateDataset):
    raw.npy        float32  (N, *patch_shape)   offset-subtracted counts
    teacher.npy    float32  (N, *patch_shape)   clipped BM4D denoising
    fg.npy         uint8    (N, *patch_shape)   foreground mask (0/1)
    brain_index.npy int32   (N,)                 index into brain_ids.json
    center.npy     int64    (N, 3)               level-0 patch centers
    offset.npy     float32  (N,)                 subtracted brain offset
    noise_params.npy float32 (N, 2)              variance slope/intercept
    brain_ids.json                              indexed source brain IDs
    transform.json                              resolved transform cfg
    config.json                                 teacher + full cache provenance

The transform cfg, teacher mode, BM4D sigma or calibrated per-brain noise
models, count dtype, and repository commit are stamped alongside the patches.
Each worker builds its datasets once (via init_datasets) so the large skeleton
arrays and cloud handles are not re-pickled per patch.

"""

import argparse
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

from aind_exaspim_image_compression.machine_learning import data_handling
from aind_exaspim_image_compression.machine_learning.noise_models import (
    load_noise_models,
)
from aind_exaspim_image_compression.machine_learning.transforms import (
    build_transform,
)
from aind_exaspim_image_compression.machine_learning.teachers import (
    TEACHER_MODES,
)
from aind_exaspim_image_compression.utils import util

# Per-split RNG stream ids so the train and val caches, even at the same base
# seed, never sample the same (brain, voxel) for a given task index.
_SEED_STREAMS = {"train": 0, "val": 1}

_WORKER_TRAIN = None
_WORKER_VAL = None
_WORKER_SEED = None
_WORKER_STREAM = 0
_WORKER_SPLIT = "train"
_COUNT_DTYPE = np.float32

# Legacy behavior remains the import-time default for callers/tests that invoke
# precompute() programmatically instead of entering the CLI block below.
teacher_mode = "raw_bm4d"
noise_models_path = None
gat_sigma_multiplier = 1.0
brain_sampling_weights = None
sampling_rois_path = None
heldout_regions_path = None
bright_sampling_weights = None


def _code_version():
    """Return the repository commit identifier when one is available."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _seed_task(index):
    """
    Seeds the global RNGs deterministically from the base seed and task index.

    Uses a SeedSequence so per-task streams are independent and well-mixed,
    making the cache reproducible and independent of worker count / task
    scheduling (executor.map assigns result i to task i regardless of which
    worker runs it). A None base seed is a no-op (nondeterministic sampling).
    """
    if _WORKER_SEED is None:
        return
    states = np.random.SeedSequence(
        [_WORKER_SEED, _WORKER_STREAM, index]
    ).generate_state(2)
    random.seed(int(states[0]))
    np.random.seed(int(states[1]))


def _init_worker(init_kwargs, base_seed, split):
    """Builds the (train, val) dataset pair per worker and caches it."""
    global _WORKER_TRAIN, _WORKER_VAL
    global _WORKER_SEED, _WORKER_STREAM, _WORKER_SPLIT
    _WORKER_SEED = base_seed
    _WORKER_SPLIT = split
    _WORKER_STREAM = _SEED_STREAMS[split]
    _WORKER_TRAIN, _WORKER_VAL = data_handling.init_datasets(**init_kwargs)


def _sample_counts(index):
    """
    Samples one count-space example for the configured split.

    Both splits build the foreground mask from the segmentation labels (used
    as-is unless segmentation_dilate > 0) unioned with the traced skeleton
    (dilated to a neurite radius). The train split does this inside
    TrainDataset; the val split draws the voxel with the same foreground-biased
    sampler, builds the annotation mask from the TrainDataset (which owns the
    segmentations and skeletons), and hands it to the ValidateDataset so the
    target and the validation metric agree.
    """
    _seed_task(index)
    if _WORKER_SPLIT == "train":
        return _WORKER_TRAIN._sample_counts()
    # sample_clean draws a patch (reading the val image and the train
    # segmentation), resampling past incoherent-artifact patches, and returns
    # the raw + labels so the val cache reads the image only once.
    brain_id, voxel, raw, labels = _WORKER_TRAIN.sample_clean(
        _WORKER_VAL.read_counts
    )
    fg_mask = _WORKER_TRAIN.annotation_mask(brain_id, voxel, labels=labels)
    return _WORKER_VAL.sample_counts(brain_id, voxel, fg_mask=fg_mask, raw=raw)


def precompute():
    if teacher_mode not in TEACHER_MODES:
        raise ValueError(
            f"unknown teacher mode {teacher_mode!r}; expected one of "
            f"{TEACHER_MODES}"
        )
    if teacher_mode == "gat_bm4d" and not noise_models_path:
        raise ValueError("gat_bm4d requires noise_models_path")
    if teacher_mode == "gat_bm4d" and (
        not np.isfinite(gat_sigma_multiplier) or gat_sigma_multiplier <= 0
    ):
        raise ValueError("gat_sigma_multiplier must be finite and positive")

    # Offset calibration would need a cloud sample the cache is meant to avoid,
    # and each worker would calibrate on its own random sample -- so the cache
    # would mix inconsistent offsets and none would match the stamped cfg. The
    # training config subtracts per-brain offsets instead; refuse the ambiguous
    # case loudly for both splits.
    if transform_cfg.get("calibrate", {}).get("offset", False):
        raise ValueError(
            "offset calibration is not supported by the cached path; bake the "
            "offset into transform_cfg or use per-brain offsets"
        )

    # Build the config each worker uses to construct its datasets. n_validate
    # is 0 (we draw validation voxels ourselves) and the transform offset stays
    # 0 because per-brain offsets are subtracted per patch.
    brain_ids = util.read_txt(brain_ids_path)
    noise_models = (
        load_noise_models(noise_models_path, required_brain_ids=brain_ids)
        if noise_models_path
        else None
    )
    sampling_rois = (
        util.read_json(sampling_rois_path) if sampling_rois_path else None
    )
    heldout_regions = (
        util.read_json(heldout_regions_path) if heldout_regions_path else None
    )
    resolved_brain_weights = brain_sampling_weights
    if resolved_brain_weights is None and noise_models is not None:
        resolved_brain_weights = {
            brain_id: model.sampling_weight
            for brain_id, model in noise_models.items()
        }
    resolved_brain_weights, brain_sampling_distribution = (
        data_handling.resolve_brain_sampling_weights(
            brain_ids, resolved_brain_weights
        )
    )
    offsets = util.read_json(offsets_path) if offsets_path else None
    if noise_models is not None:
        calibrated_offsets = {
            brain_id: model.offset for brain_id, model in noise_models.items()
        }
        if offsets is None:
            offsets = calibrated_offsets
        else:
            for brain_id in brain_ids:
                if not np.isclose(
                    offsets[str(brain_id)], calibrated_offsets[str(brain_id)]
                ):
                    raise ValueError(
                        f"offset for brain {brain_id!r} disagrees between "
                        "offsets_path and noise_models_path"
                    )
    resolved_transform_cfg = build_transform(transform_cfg).cfg
    init_kwargs = dict(
        brain_ids=brain_ids,
        img_paths_json=img_prefixes_path,
        patch_shape=patch_shape,
        foreground_sampling_rate=foreground_sampling_rate,
        min_foreground_voxels=min_foreground_voxels,
        min_segmentation_volume=min_segmentation_volume,
        n_validate_examples=0,
        offsets=offsets,
        reject_incoherent_patches=reject_incoherent_patches,
        coherence_min_autocorr=coherence_min_autocorr,
        coherence_max_highfreq_frac=coherence_max_highfreq_frac,
        coherence_min_segment_voxels=coherence_min_segment_voxels,
        coherence_smooth_sigma=coherence_smooth_sigma,
        coherence_lag=coherence_lag,
        max_resample_attempts=max_resample_attempts,
        segmentation_prefixes_path=segmentation_prefixes_path,
        segmentation_dilate=segmentation_dilate,
        sigma_bm4d=sigma_bm4d,
        teacher_mode=teacher_mode,
        noise_models=noise_models,
        gat_sigma_multiplier=gat_sigma_multiplier,
        brain_sampling_weights=resolved_brain_weights,
        sampling_rois=sampling_rois,
        heldout_regions=heldout_regions,
        exclude_heldout=(split == "train"),
        bright_sampling_weights=bright_sampling_weights,
        skeleton_radius=skeleton_radius,
        swc_pointers=swc_pointers,
        transform_cfg=resolved_transform_cfg,
    )

    # Pre-allocate memory-mapped outputs and stream results into them.
    util.mkdir(cache_dir)
    util.write_json(
        f"{cache_dir}/config.json",
        {
            "split": split,
            "cache_dir": cache_dir,
            "n_patches": n_patches,
            "brain_ids_path": brain_ids_path,
            "img_prefixes_path": img_prefixes_path,
            "segmentation_prefixes_path": segmentation_prefixes_path,
            "offsets_path": offsets_path,
            "swc_pointers": swc_pointers,
            "transform_cfg": resolved_transform_cfg,
            "foreground_sampling_rate": foreground_sampling_rate,
            "min_foreground_voxels": min_foreground_voxels,
            "min_segmentation_volume": min_segmentation_volume,
            "patch_shape": patch_shape,
            "skeleton_radius": skeleton_radius,
            "segmentation_dilate": segmentation_dilate,
            "sigma_bm4d": (
                sigma_bm4d if teacher_mode == "raw_bm4d" else None
            ),
            "teacher_mode": teacher_mode,
            "gat_sigma_multiplier": gat_sigma_multiplier,
            "brain_sampling_weights": resolved_brain_weights,
            "brain_sampling_distribution": brain_sampling_distribution,
            "sampling_rois_path": sampling_rois_path,
            "sampling_rois": sampling_rois,
            "bright_sampling_weights": bright_sampling_weights,
            "heldout_regions_path": heldout_regions_path,
            "heldout_regions": heldout_regions,
            "exclude_heldout": split == "train",
            "noise_models_path": noise_models_path,
            "noise_models": (
                {
                    brain_id: model.to_dict()
                    for brain_id, model in sorted(noise_models.items())
                }
                if noise_models is not None
                else None
            ),
            "saturation_margins": (
                {
                    brain_id: model.saturation_margin
                    for brain_id, model in sorted(noise_models.items())
                }
                if noise_models is not None
                else None
            ),
            "reject_incoherent_patches": reject_incoherent_patches,
            "coherence_min_autocorr": coherence_min_autocorr,
            "coherence_max_highfreq_frac": coherence_max_highfreq_frac,
            "coherence_min_segment_voxels": (
                coherence_min_segment_voxels
            ),
            "coherence_smooth_sigma": coherence_smooth_sigma,
            "coherence_lag": coherence_lag,
            "max_resample_attempts": max_resample_attempts,
            "seed": seed,
            "seed_stream": _SEED_STREAMS[split],
            "num_workers": num_workers,
            "count_dtype": np.dtype(_COUNT_DTYPE).name,
            "code_version": _code_version(),
            "cache_metadata_version": 1,
        },
    )
    util.write_json(
        f"{cache_dir}/brain_ids.json",
        [str(brain_id) for brain_id in brain_ids],
    )
    shape = (n_patches,) + tuple(patch_shape)
    raw_mm = open_memmap(
        f"{cache_dir}/raw.npy", mode="w+", dtype=_COUNT_DTYPE, shape=shape
    )
    teacher_mm = open_memmap(
        f"{cache_dir}/teacher.npy", mode="w+", dtype=_COUNT_DTYPE, shape=shape
    )
    fg_mm = open_memmap(
        f"{cache_dir}/fg.npy", mode="w+", dtype=np.uint8, shape=shape
    )
    brain_index_mm = open_memmap(
        f"{cache_dir}/brain_index.npy",
        mode="w+",
        dtype=np.int32,
        shape=(n_patches,),
    )
    center_mm = open_memmap(
        f"{cache_dir}/center.npy",
        mode="w+",
        dtype=np.int64,
        shape=(n_patches, 3),
    )
    offset_mm = open_memmap(
        f"{cache_dir}/offset.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_patches,),
    )
    noise_params_mm = open_memmap(
        f"{cache_dir}/noise_params.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_patches, 2),
    )

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(init_kwargs, seed, split),
    ) as executor:
        results = executor.map(
            _sample_counts, range(n_patches), chunksize=1
        )
        for i, record in enumerate(
            tqdm(results, total=n_patches, desc=f"Precompute ({split})")
        ):
            raw_mm[i] = np.asarray(record["raw"], dtype=_COUNT_DTYPE)
            teacher_mm[i] = np.asarray(
                record["teacher"], dtype=_COUNT_DTYPE
            )
            fg_mm[i] = np.asarray(record["foreground"], dtype=np.uint8)
            brain_index_mm[i] = np.int32(record["brain_index"])
            center_mm[i] = np.asarray(record["center"], dtype=np.int64)
            offset_mm[i] = np.float32(record["offset"])
            noise_params_mm[i] = np.asarray(
                record["noise_params"], dtype=np.float32
            )

    raw_mm.flush()
    teacher_mm.flush()
    fg_mm.flush()
    brain_index_mm.flush()
    center_mm.flush()
    offset_mm.flush()
    noise_params_mm.flush()

    # Stamp the resolved transform cfg so training rebuilds it exactly without
    # touching the cloud.
    util.write_json(
        f"{cache_dir}/transform.json", resolved_transform_cfg
    )
    print(f"Wrote {n_patches} {split} patches to {cache_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Which cache to build (default: train).",
    )
    split = parser.parse_args().split

    # Paths (shared by both splits)
    brain_ids_path = "/data/train_brain_ids.txt"
    img_prefixes_path = "/data/exaspim_image_prefixes.json"
    segmentation_prefixes_path = "/data/exaspim_segmentation_prefixes.json"
    offsets_path = "/data/exaspim_background_offsets.json"
    noise_models_path = None
    sampling_rois_path = None
    heldout_regions_path = None

    # SWC pointer (shared)
    swc_pointers = {
        "bucket_name": "allen-nd-goog",
        "path": "ground_truth_tracings",
    }

    # Transform cfg (shared; offset 0, per-brain offsets subtracted per patch).
    # Only max_count is used here, to clip the BM4D teacher. Keeping this
    # shared is the point of one script: the train and val caches must use the
    # identical mapping or the model trains and validates under different
    # transforms.
    transform_cfg = {
        "kind": "asinh",
        "params": {"offset": 0.0, "scale": 32.0},
    }

    # Teacher experiment. raw_bm4d preserves the existing count-space teacher.
    # For gat_bm4d, point noise_models_path at estimate_noise_models.py output;
    # its per-brain offsets are used when offsets_path is None. Values above 1
    # for gat_sigma_multiplier intentionally smooth more than the fitted noise.
    teacher_mode = "raw_bm4d"
    gat_sigma_multiplier = 1.0

    # Sampling weights default to NoiseModel.sampling_weight when calibrated
    # models are loaded. Explicit values override that source. A bright-brain
    # mixture can deliberately balance saturated cores, unsaturated halos,
    # bright unsaturated signal, and ordinary background.
    brain_sampling_weights = None
    bright_sampling_weights = None
    # Example targeting roughly one-third bright-brain patches among ten
    # brains, with a deliberate within-brain condition mixture:
    # brain_sampling_weights = {"734348": 4.0}
    # bright_sampling_weights = {
    #     "734348": {
    #         "saturated_core": 0.30,
    #         "halo": 0.25,
    #         "bright_unsaturated": 0.20,
    #         "background": 0.25,
    #     }
    # }

    # Sampling / patch parameters (shared)
    foreground_sampling_rate = 0.5
    min_foreground_voxels = 50
    min_segmentation_volume = 200
    patch_shape = (64, 64, 64)
    # Neurite radius (voxels) the traced skeleton is dilated to in the mask.
    skeleton_radius = 2
    # Dilation (voxels) applied to the segmentation labels; 0 uses them as-is,
    # since the labels already mark neurite voxels.
    segmentation_dilate = 0
    sigma_bm4d = 24

    # Reject whole patches contaminated by a bright, spatially incoherent
    # raw-image processing artifact (blocky salt-and-pepper noise) the FFN
    # mislabels as a neurite. The artifact corrupts the raw input itself, so
    # such a patch is a poor training example even with the label removed;
    # sample_clean discards it and resamples (before BM4D, so rejects are
    # cheap). A segment triggers rejection only when it fails BOTH tests --
    # lag-2 autocorrelation below coherence_min_autocorr AND high-frequency
    # energy fraction above coherence_max_highfreq_frac -- so dim-but-smooth
    # neurites do not. Lag 2 (not 1) is the discriminating scale: the brightest
    # artifacts correlate at lag 1 but decorrelate by lag 2, while real
    # PSF-blurred signal stays correlated. Only segments >=
    # coherence_min_segment_voxels are scored. See
    # metrics.patch_has_incoherent_segment.
    reject_incoherent_patches = True
    coherence_min_autocorr = 0.4
    coherence_max_highfreq_frac = 0.35
    coherence_min_segment_voxels = 50
    coherence_smooth_sigma = 1.0
    coherence_lag = 2
    # Give up resampling a clean patch after this many artifact hits and accept
    # the last draw (rare; keeps the fixed-size cache build from stalling).
    max_resample_attempts = 50

    # Base RNG seed for reproducibility: with a fixed seed the sampled pool is
    # identical across runs and independent of num_workers. Set to None for
    # nondeterministic sampling. num_workers=None uses all CPUs.
    seed = 42
    num_workers = None

    # Per-split output location and pool size.
    if split == "train":
        # ~2.4 MB/patch (fp32 raw+teacher + uint8 fg), so 30000 ~= 71 GB.
        cache_dir = "/results/patch_cache"
        n_patches = 30000
    else:
        # The per-patch metrics are heavy-tailed (a sizable fraction of patches
        # are near-pure background that compress at hundreds of x), so a small
        # set makes the reported median cratio -- and thus checkpoint selection
        # -- noisy. ~500 keeps that sampling error small; returns diminish past
        # ~1000. Disk is cheap.
        cache_dir = "/results/val_patch_cache"
        n_patches = 500

    precompute()
