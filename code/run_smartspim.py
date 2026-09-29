"""
Precompute a SmartSPIM patch cache to disk so training is GPU-bound.

Same pipeline as run.py but configured for SmartSPIM data, which has no
segmentation labels and no SWC skeletons. Foreground comes from detected
cells alone, so none of the segmentation/skeleton machinery in
data_handling (label masks, skeleton rasterization, incoherent-segment
rejection) is wired up here; run.py still uses all of it for exaSPIM.

Foreground patches are centered on detected cells (jittered) for brains whose
stitched asset carries a cell segmentation on the image channel, at
<asset>/image_cell_segmentation/<channel>/detected_cells.csv. That mapping is
resolved at startup and written to /results/smartspim_cell_prefixes.json;
brains without one (no segmentation, or one on a different channel) fall back
to intensity-threshold foreground sampling. Foreground masks are the robust
intensity threshold throughout, for both splits.

Reads per-brain background offsets from
/data/smartspim_background_offsets.json (written by
estimate_background_offsets.py --img_prefixes_path
/data/smartspim_image_prefixes.json).

Outputs, under cache_dir (identical layout for both splits, so the val cache
loads with CachedValidateDataset):
    raw.npy        float32  (N, *patch_shape)   offset-subtracted counts
    teacher.npy    float32  (N, *patch_shape)   clipped BM4D denoising
    fg.npy         uint8    (N, *patch_shape)   foreground mask (0/1)
    transform.json                              resolved transform cfg
    config.json                                 full precompute configuration

"""

import random
import shutil

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from numpy.lib.format import open_memmap
from tqdm import tqdm

from aind_exaspim_image_compression.machine_learning import data_handling
from aind_exaspim_image_compression.machine_learning.transforms import (
    build_transform,
)
from aind_exaspim_image_compression.utils import util

_SEED_STREAMS = {"train": 0, "val": 1}

_WORKER_TRAIN = None
_WORKER_VAL = None
_WORKER_SEED = None
_WORKER_STREAM = 0
_WORKER_SPLIT = "train"
_COUNT_DTYPE = np.float32


def _seed_task(index):
    if _WORKER_SEED is None:
        return
    states = np.random.SeedSequence(
        [_WORKER_SEED, _WORKER_STREAM, index]
    ).generate_state(2)
    random.seed(int(states[0]))
    np.random.seed(int(states[1]))


def _init_worker(init_kwargs, base_seed, split):
    global _WORKER_TRAIN, _WORKER_VAL
    global _WORKER_SEED, _WORKER_STREAM, _WORKER_SPLIT
    _WORKER_SEED = base_seed
    _WORKER_SPLIT = split
    _WORKER_STREAM = _SEED_STREAMS[split]
    _WORKER_TRAIN, _WORKER_VAL = data_handling.init_datasets(**init_kwargs)


def _sample_counts(index):
    """
    Samples one count-space example for the split this worker serves.

    The validation split draws its voxel from the train dataset (which owns
    the detected cells) but reads the image through the validation dataset.
    No foreground mask is passed: with no segmentation or skeletons, the mask
    is the intensity threshold on both sides, so training and validation
    masks already agree.
    """
    _seed_task(index)
    if _WORKER_SPLIT == "train":
        return _WORKER_TRAIN._sample_counts()
    brain_id = _WORKER_TRAIN.sample_brain()
    voxel = _WORKER_TRAIN.sample_voxel(brain_id)
    raw = _WORKER_VAL.read_counts(brain_id, voxel)
    return _WORKER_VAL.sample_counts(brain_id, voxel, raw=raw)


def precompute(cache_dir, n_patches, split):
    if transform_cfg.get("calibrate", {}).get("offset", False):
        raise ValueError(
            "offset calibration is not supported by the cached path; bake the "
            "offset into transform_cfg or use per-brain offsets"
        )

    brain_ids = util.read_txt(brain_ids_path)
    offsets = util.read_json(offsets_path) if offsets_path else None
    resolved_transform_cfg = build_transform(transform_cfg).cfg
    init_kwargs = dict(
        brain_ids=brain_ids,
        img_paths_json=img_prefixes_path,
        patch_shape=patch_shape,
        boundary_buffer=boundary_buffer,
        cell_prefixes_path=cell_prefixes_path,
        cell_max_shift=cell_max_shift,
        foreground_sampling_rate=foreground_sampling_rate,
        min_foreground_voxels=min_foreground_voxels,
        n_validate_examples=0,
        offsets=offsets,
        sigma_bm4d=sigma_bm4d,
        transform_cfg=resolved_transform_cfg,
    )

    util.mkdir(cache_dir)
    util.write_json(
        f"{cache_dir}/config.json",
        {
            "split": split,
            "cache_dir": cache_dir,
            "n_patches": n_patches,
            "brain_ids_path": brain_ids_path,
            "img_prefixes_path": img_prefixes_path,
            "cell_prefixes_path": cell_prefixes_path,
            "cell_max_shift": cell_max_shift,
            "offsets_path": offsets_path,
            "transform_cfg": resolved_transform_cfg,
            "foreground_sampling_rate": foreground_sampling_rate,
            "min_foreground_voxels": min_foreground_voxels,
            "patch_shape": patch_shape,
            "boundary_buffer": boundary_buffer,
            "sigma_bm4d": sigma_bm4d,
            "seed": seed,
            "seed_stream": _SEED_STREAMS[split],
            "num_workers": num_workers,
            "count_dtype": np.dtype(_COUNT_DTYPE).name,
        },
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
            raw_mm[i] = np.asarray(raw, dtype=_COUNT_DTYPE)
            teacher_mm[i] = np.asarray(teacher, dtype=_COUNT_DTYPE)
            fg_mm[i] = np.asarray(fg, dtype=np.uint8)

    raw_mm.flush()
    teacher_mm.flush()
    fg_mm.flush()

    util.write_json(f"{cache_dir}/transform.json", resolved_transform_cfg)
    print(f"Wrote {n_patches} {split} patches to {cache_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Precompute SmartSPIM patch cache"
    )
    parser.add_argument(
        "--sigma_bm4d",
        type=int,
        default=24,
        help="BM4D noise level used for teacher denoising (default: 24)",
    )
    args = parser.parse_args()

    brain_ids_path = "/data/train_smartspim_brain_ids.txt"

    # Save brain ids in results
    shutil.copy2(brain_ids_path, "/results/train_smartspim_brain_ids.txt")

    # Paths
    img_prefixes_path = "/data/smartspim_image_prefixes.json"

    # Resolved below from the image prefixes, then written to /results so the
    # exact mapping used is recorded alongside the cache. Set
    # use_detected_cells to False to fall back to intensity-threshold
    # foreground sampling for every brain.
    use_detected_cells = True
    cell_prefixes_path = "/results/smartspim_cell_prefixes.json"
    offsets_path = "/data/smartspim_background_offsets.json"

    # Transform cfg (offset 0; per-brain offsets subtracted per patch)
    transform_cfg = {
        "kind": "asinh",
        "params": {"offset": 0.0, "scale": 32.0},
    }

    # Sampling / patch parameters. foreground_sampling_rate splits patches
    # between cell-centered and random interior (background) locations; the
    # denoiser needs both. min_foreground_voxels only applies to brains with
    # no detected cells, which fall back to intensity-threshold sampling.
    foreground_sampling_rate = 0.5
    min_foreground_voxels = 50
    patch_shape = (64, 64, 64)
    boundary_buffer = 128  # SmartSPIM z is ~4k voxels; run.py's 5000 exceeds it
    cell_max_shift = None  # defaults to patch_shape // 4
    sigma_bm4d = args.sigma_bm4d

    seed = 42
    num_workers = None

    # Locate each brain's detected_cells.csv (sibling of the fused image,
    # same channel) and record the mapping next to the cache.
    if use_detected_cells:
        cell_paths = data_handling.find_detected_cells_paths(
            util.read_txt(brain_ids_path), img_prefixes_path
        )
        util.write_json(cell_prefixes_path, cell_paths)
        print(f"Found detected cells for {len(cell_paths)} brains")
    else:
        cell_prefixes_path = None

    # Train cached dataset
    train_cache_dir = f"/results/smartspim_patch_cache_sigma{sigma_bm4d}"
    n_train_patches = 30000
    precompute(train_cache_dir, n_train_patches, "train")

    # Val cached dataset
    val_cache_dir = f"/results/smartspim_val_patch_cache_sigma{sigma_bm4d}"
    n_val_patches = 500
    precompute(val_cache_dir, n_val_patches, "val")
