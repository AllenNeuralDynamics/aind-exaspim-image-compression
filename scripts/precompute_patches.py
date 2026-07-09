"""
Precompute a pool of training patches to disk so training is GPU-bound.

The training bottleneck is per-patch BM4D + cloud reads on the CPU, which
leaves the GPU idle. This script samples a fixed pool of patches once and
writes the expensive count-space intermediates -- (raw with the per-brain
offset subtracted, clipped BM4D teacher, foreground mask) -- to memory-mapped
arrays. Training then reads a random cached patch and applies only the cheap
transform + target construction (see CachedPatchDataset).

Outputs, under cache_dir:
    raw.npy      float16  (N, *patch_shape)   offset-subtracted counts
    teacher.npy  float16  (N, *patch_shape)   clipped BM4D denoising
    fg.npy       uint8    (N, *patch_shape)   foreground mask (0/1)

Each worker builds its own dataset once (via init_datasets) so the large
skeleton arrays and cloud handles are not re-pickled per patch.

"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from numpy.lib.format import open_memmap
from tqdm import tqdm

from aind_exaspim_image_compression.machine_learning import data_handling
from aind_exaspim_image_compression.utils import util

_WORKER_DATASET = None


def _init_worker(init_kwargs):
    """Builds one TrainDataset per worker process and caches it globally."""
    global _WORKER_DATASET
    _WORKER_DATASET, _ = data_handling.init_datasets(**init_kwargs)


def _sample_counts(_):
    """Samples one count-space example from the per-worker dataset."""
    return _WORKER_DATASET._sample_counts()


def _to_float16(arr):
    """Clips to the float16 range before casting (avoids inf at saturation)."""
    return np.clip(arr, -65504, 65504).astype(np.float16)


def precompute():
    # Build the config each worker uses to construct its dataset. n_validate
    # is 0 (we only need training sampling) and the transform offset stays 0
    # because per-brain offsets are subtracted in _sample_counts.
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
        initargs=(init_kwargs,),
    ) as executor:
        results = executor.map(
            _sample_counts, range(n_patches), chunksize=1
        )
        for i, (raw, teacher, fg) in enumerate(
            tqdm(results, total=n_patches, desc="Precompute")
        ):
            raw_mm[i] = _to_float16(raw)
            teacher_mm[i] = _to_float16(teacher)
            fg_mm[i] = np.asarray(fg, dtype=np.uint8)

    raw_mm.flush()
    teacher_mm.flush()
    fg_mm.flush()
    print(f"Wrote {n_patches} patches to {cache_dir}")


if __name__ == "__main__":
    # Paths (match train_bm4dnet.py)
    brain_ids_path = "/data/train_brain_ids.txt"
    img_prefixes_path = "/data/exaspim_image_prefixes.json"
    segmentation_prefixes_path = "/data/exaspim_segmentation_prefixes.json"
    offsets_path = "/data/exaspim_background_offsets.json"
    cache_dir = "/results/patch_cache"

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

    # Pool size and parallelism. ~1.3 MB/patch (fp16 raw+teacher + uint8 fg),
    # so 8000 patches ~= 10 GB. num_workers=None uses all CPUs.
    n_patches = 30000
    num_workers = None

    precompute()
