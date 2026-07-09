"""
Precompute a fixed pool of validation patches to disk.

The training run's other startup cost (besides the training pool) is the live
validation set: init_datasets samples a handful of voxels and, for each, does
a cloud read + a serial BM4D denoising on the CPU while the GPU sits idle.
This script does that work once, offline, and writes the expensive count-space
intermediates -- (raw with the per-brain offset subtracted, clipped BM4D
teacher, foreground mask) -- to memory-mapped arrays. A cache-backed training
run then reads them via CachedValidateDataset and applies only the cheap
transform + target construction, so no cloud access or BM4D happens at startup.

Voxels are drawn exactly as init_datasets draws them for validation: the
foreground-biased sampler on the TrainDataset (which needs the skeletons /
segmentations), while the count-space example -- crucially the intensity-only
foreground mask used by the validation metric split -- is computed by the
ValidateDataset. Both are built once per worker via init_datasets.

Outputs, under cache_dir (same layout as the training cache, so it loads with
CachedValidateDataset):
    raw.npy        float16  (N, *patch_shape)   offset-subtracted counts
    teacher.npy    float16  (N, *patch_shape)   clipped BM4D denoising
    fg.npy         uint8    (N, *patch_shape)   foreground mask (0/1)
    transform.json                              resolved transform cfg

The transform cfg is stamped alongside the patches so the training run can
construct the identical transform without touching the cloud.

"""

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

_WORKER_TRAIN = None
_WORKER_VAL = None
_WORKER_SEED = None

# Distinct RNG stream id for this script so the validation and training caches,
# even at the same base seed, never sample the same (brain, voxel) for a given
# task index. Keep it different from the training precompute's stream.
_SEED_STREAM = 1


def _seed_task(base_seed, index):
    """
    Seeds the global RNGs deterministically from a base seed and task index.

    Uses a SeedSequence so per-task streams are independent and well-mixed,
    making the cache reproducible and independent of worker count / task
    scheduling (executor.map assigns result i to task i regardless of which
    worker runs it). A None base seed is a no-op (nondeterministic sampling).
    """
    if base_seed is None:
        return
    states = np.random.SeedSequence(
        [base_seed, _SEED_STREAM, index]
    ).generate_state(2)
    random.seed(int(states[0]))
    np.random.seed(int(states[1]))


def _init_worker(init_kwargs, base_seed):
    """Builds one (train, val) dataset pair per worker and caches it."""
    global _WORKER_TRAIN, _WORKER_VAL, _WORKER_SEED
    _WORKER_SEED = base_seed
    _WORKER_TRAIN, _WORKER_VAL = data_handling.init_datasets(**init_kwargs)


def _sample_val_counts(index):
    """Samples one validation count-space example from the per-worker pair.

    The voxel is drawn by the TrainDataset's foreground-biased sampler (as in
    init_datasets); the count-space example, including the intensity-only mask,
    is produced by the ValidateDataset so it matches the metric split.
    """
    _seed_task(_WORKER_SEED, index)
    brain_id = _WORKER_TRAIN.sample_brain()
    voxel = _WORKER_TRAIN.sample_voxel(brain_id)
    return _WORKER_VAL.sample_counts(brain_id, voxel)


def _to_float16(arr):
    """Clips to the float16 range before casting (avoids inf at saturation)."""
    return np.clip(arr, -65504, 65504).astype(np.float16)


def precompute():
    # Offset calibration would need a cloud sample that this cache is meant to
    # avoid; the training config subtracts per-brain offsets instead, so the
    # transform offset stays fixed. Refuse the ambiguous case loudly.
    if transform_cfg.get("calibrate", {}).get("offset", False):
        raise ValueError(
            "offset calibration is not supported by the cached path; bake the "
            "offset into transform_cfg or use per-brain offsets"
        )

    # Build the config each worker uses to construct its datasets. n_validate
    # is 0 (we draw validation voxels ourselves) and the transform offset stays
    # 0 because per-brain offsets are subtracted in sample_counts.
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
        initargs=(init_kwargs, seed),
    ) as executor:
        results = executor.map(
            _sample_val_counts, range(n_patches), chunksize=1
        )
        for i, (raw, teacher, fg) in enumerate(
            tqdm(results, total=n_patches, desc="Precompute (val)")
        ):
            raw_mm[i] = _to_float16(raw)
            teacher_mm[i] = _to_float16(teacher)
            fg_mm[i] = np.asarray(fg, dtype=np.uint8)

    raw_mm.flush()
    teacher_mm.flush()
    fg_mm.flush()

    # Stamp the resolved transform cfg so training rebuilds it exactly without
    # touching the cloud.
    util.write_json(f"{cache_dir}/transform.json", build_transform(transform_cfg).cfg)
    print(f"Wrote {n_patches} validation patches to {cache_dir}")


if __name__ == "__main__":
    # Paths (match train_bm4dnet.py)
    brain_ids_path = "/data/train_brain_ids.txt"
    img_prefixes_path = "/data/exaspim_image_prefixes.json"
    segmentation_prefixes_path = "/data/exaspim_segmentation_prefixes.json"
    offsets_path = "/data/exaspim_background_offsets.json"
    cache_dir = "/results/val_patch_cache"

    # SWC pointer
    swc_pointers = {
        "bucket_name": "allen-nd-goog",
        "path": "ground_truth_tracings",
    }

    # Transform cfg (offset 0; per-brain offsets are subtracted per patch).
    # Only max_count is used here, to clip the BM4D teacher.
    transform_cfg = {
        "kind": "asinh",
        "params": {"offset": 0.0, "scale": 32.0},
    }

    # Sampling / patch parameters (match training)
    foreground_sampling_rate = 0.5
    min_foreground_voxels = 50
    min_segmentation_volume = 200
    patch_shape = (64, 64, 64)
    preserve_foreground = True
    sigma_bm4d = 24

    # Pool size and parallelism. The per-patch metrics are heavy-tailed (a
    # sizable fraction of patches are near-pure background that compress at
    # hundreds of x), so a small set makes the reported median cratio -- and
    # thus checkpoint selection -- noisy. ~500 patches keeps the sampling error
    # of the selection score small; returns diminish past ~1000. Disk is cheap
    # (~1.3 MB/patch). num_workers=None uses all CPUs.
    n_patches = 500
    num_workers = None

    # Base RNG seed for reproducibility: with a fixed seed the sampled set is
    # identical across runs and independent of num_workers. Set to None for
    # nondeterministic sampling. A distinct RNG stream (see _SEED_STREAM) keeps
    # these validation patches off the training patches at the same seed.
    seed = 42

    precompute()
