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
build the foreground mask from the segmentation labels unioned with the traced
skeleton (each dilated), so the training target and the validation metric agree
on what counts as neurite signal -- bright non-neuronal structures (noise,
off-target label) are left for the BM4D teacher to denoise rather than
preserved, while neurites the segmentation misses are still protected by the
skeleton. The train split builds the mask inside TrainDataset; the val split
builds the annotation mask from the TrainDataset and hands it to the
ValidateDataset. The splits otherwise differ only in the outputs each records.

A distinct RNG stream per split means the two caches never sample the same
(brain, voxel) for a given task index when built with the same base seed.

Outputs, under cache_dir (identical layout for both splits, so the val cache
loads with CachedValidateDataset):
    raw.npy        float16  (N, *patch_shape)   offset-subtracted counts
    teacher.npy    float16  (N, *patch_shape)   clipped BM4D denoising
    fg.npy         uint8    (N, *patch_shape)   foreground mask (0/1)
    transform.json                              resolved transform cfg

The transform cfg is stamped alongside the patches so the training run rebuilds
the identical transform without touching the cloud. Each worker builds its
datasets once (via init_datasets) so the large skeleton arrays and cloud
handles are not re-pickled per patch.

"""

import argparse
import random

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from numpy.lib.format import open_memmap
from tqdm import tqdm

from aind_exaspim_image_compression.machine_learning import data_handling
from aind_exaspim_image_compression.machine_learning.transforms import (
    build_transform,
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

    Both splits build the foreground mask from the segmentation labels unioned
    with the traced skeleton (each dilated). The train split does this inside
    TrainDataset; the val split draws the voxel with the same foreground-biased
    sampler, builds the annotation mask from the TrainDataset (which owns the
    segmentations and skeletons), and hands it to the ValidateDataset so the
    target and the validation metric agree.
    """
    _seed_task(index)
    if _WORKER_SPLIT == "train":
        return _WORKER_TRAIN._sample_counts()
    brain_id = _WORKER_TRAIN.sample_brain()
    voxel = _WORKER_TRAIN.sample_voxel(brain_id)
    fg_mask = _WORKER_TRAIN.annotation_mask(brain_id, voxel)
    return _WORKER_VAL.sample_counts(brain_id, voxel, fg_mask=fg_mask)


def _to_float16(arr):
    """Clips to the float16 range before casting (avoids inf at saturation)."""
    return np.clip(arr, -65504, 65504).astype(np.float16)


def precompute():
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
    offsets = util.read_json(offsets_path) if offsets_path else None
    init_kwargs = dict(
        brain_ids=brain_ids,
        img_paths_json=img_prefixes_path,
        patch_shape=patch_shape,
        foreground_sampling_rate=foreground_sampling_rate,
        min_foreground_voxels=min_foreground_voxels,
        min_segmentation_volume=min_segmentation_volume,
        n_validate_examples=0,
        offsets=offsets,
        preserve_foreground=preserve_foreground,
        segmentation_prefixes_path=segmentation_prefixes_path,
        sigma_bm4d=sigma_bm4d,
        skeleton_radius=skeleton_radius,
        swc_pointers=swc_pointers,
        transform_cfg=transform_cfg,
    )

    # Pre-allocate memory-mapped outputs and stream results into them.
    util.mkdir(cache_dir)
    shape = (n_patches,) + tuple(patch_shape)
    raw_mm = open_memmap(
        f"{cache_dir}/raw.npy", mode="w+", dtype=np.float16, shape=shape
    )
    teacher_mm = open_memmap(
        f"{cache_dir}/teacher.npy", mode="w+", dtype=np.float16, shape=shape
    )
    fg_mm = open_memmap(
        f"{cache_dir}/fg.npy", mode="w+", dtype=np.uint8, shape=shape
    )

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(init_kwargs, seed, split),
    ) as executor:
        results = executor.map(
            _sample_counts, range(n_patches), chunksize=1
        )
        for i, (raw, teacher, fg) in enumerate(
            tqdm(results, total=n_patches, desc=f"Precompute ({split})")
        ):
            raw_mm[i] = _to_float16(raw)
            teacher_mm[i] = _to_float16(teacher)
            fg_mm[i] = np.asarray(fg, dtype=np.uint8)

    raw_mm.flush()
    teacher_mm.flush()
    fg_mm.flush()

    # Stamp the resolved transform cfg so training rebuilds it exactly without
    # touching the cloud.
    util.write_json(
        f"{cache_dir}/transform.json", build_transform(transform_cfg).cfg
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

    # Sampling / patch parameters (shared)
    foreground_sampling_rate = 0.5
    min_foreground_voxels = 50
    min_segmentation_volume = 200
    patch_shape = (64, 64, 64)
    # Neurite radius (voxels) the traced skeleton is dilated to in the mask.
    skeleton_radius = 2
    preserve_foreground = True
    sigma_bm4d = 24

    # Base RNG seed for reproducibility: with a fixed seed the sampled pool is
    # identical across runs and independent of num_workers. Set to None for
    # nondeterministic sampling. num_workers=None uses all CPUs.
    seed = 42
    num_workers = None

    # Per-split output location and pool size.
    if split == "train":
        # ~1.3 MB/patch (fp16 raw+teacher + uint8 fg), so 30000 ~= 40 GB.
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
