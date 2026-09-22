"""
Sanity check that CellLocationsDataset samples patches at valid cell locations.

Loads a random subset of SmartSPIM brains from S3 using
smartspim_image_prefixes.json, ingests each brain's image and detected-cell
CSV, samples N patches near cell locations, and saves XY/XZ/YZ
max-intensity projections to --output_dir.

Patches are read raw (no BM4D) since the goal is to verify that voxel
coordinates land on real signal, not to produce denoised examples.

Usage
-----
    python scripts/test_smartspim_sampling.py
    python scripts/test_smartspim_sampling.py --n_patches 16 --n_brains 2
    python scripts/test_smartspim_sampling.py --output_dir /tmp/mips
"""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aind_exaspim_image_compression.machine_learning.data_handling import (
    CellLocationsDataset,
)
from aind_exaspim_image_compression.utils import util

# SmartSPIM full-resolution voxel size in (x, y, z) µm/voxel
SMARTSPIM_ANISOTROPY = (1.8, 1.8, 2.0)
PATCH_SHAPE = (64, 64, 64)


def get_csv_path(zarr_path):
    """Return detected_cells.csv path derived from the zarr image path.

    The channel name (e.g. Ex_561_Em_593) is extracted from the zarr path
    itself so it stays consistent with the image channel.
    """
    base = zarr_path.split("/image_tile_fusing/")[0]
    channel = zarr_path.split("/OMEZarr/")[1].removesuffix(".zarr")
    return f"{base}/image_cell_segmentation/{channel}/detected_cells.csv"


def save_mips(patch, brain_id, voxel, out_path):
    """Save XY, XZ, YZ max-intensity projections of a 3D patch."""
    vmax = float(np.percentile(patch, 99.9)) or 1.0
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    for i, name in enumerate(["XY", "XZ", "YZ"]):
        axs[i].imshow(np.max(patch, axis=i), vmax=vmax, cmap="gray")
        axs[i].set_title(name, fontsize=13)
        axs[i].axis("off")
    fig.suptitle(f"brain {brain_id}  voxel {voxel}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Sample SmartSPIM patches at cell locations and save MIPs."
    )
    parser.add_argument(
        "--n_patches", type=int, default=32,
        help="Number of patches to sample (default: 32).",
    )
    parser.add_argument(
        "--n_brains", type=int, default=3,
        help="Number of brains to load from the JSON (default: 3).",
    )
    parser.add_argument(
        "--prefixes_json",
        default=os.path.join(
            os.path.dirname(__file__), "..", "data",
            "smartspim_image_prefixes.json",
        ),
        help="Path to smartspim_image_prefixes.json.",
    )
    parser.add_argument(
        "--output_dir", default="/results/sanity_check_sampling",
        help="Directory to write MIP PNGs (default: /results/sanity_check_sampling).",
    )
    parser.add_argument(
        "--patch_shape", type=int, nargs=3, default=list(PATCH_SHAPE),
        metavar=("Z", "Y", "X"),
        help="Patch shape in voxels (default: 64 64 64).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load prefix map and shuffle; iterate until n_brains are ingested so
    # brains without a detected-cells CSV don't count against the quota.
    prefixes = util.read_json(args.prefixes_json)
    shuffled = list(prefixes.keys())
    random.shuffle(shuffled)

    patch_shape = tuple(args.patch_shape)
    dataset = CellLocationsDataset(
        patch_shape=patch_shape,
        anisotropy=SMARTSPIM_ANISOTROPY,
        n_examples_per_epoch=args.n_patches,
    )

    ingested = []
    for brain_id in shuffled:
        if len(ingested) >= args.n_brains:
            break
        zarr_path = prefixes[brain_id]
        img_path = zarr_path + "/0"
        csv_path = get_csv_path(zarr_path)
        print(f"Ingesting brain {brain_id} ...")
        print(f"  img : {img_path}")
        print(f"  csv : {csv_path}")
        try:
            dataset.ingest_brain(brain_id, img_path, csv_path)
            ingested.append(brain_id)
        except FileNotFoundError:
            print(f"  [skip] no detected_cells.csv for brain {brain_id}")

    if not ingested:
        print("No brains could be ingested — exiting.")
        return
    print(f"Successfully ingested: {ingested}")

    print(f"\nSampling {args.n_patches} patches ...")
    for i in range(args.n_patches):
        brain_id = dataset.sample_brain()
        voxel = dataset.sample_voxel(brain_id)
        patch = np.asarray(
            dataset.read_patch(brain_id, voxel), dtype=np.float32
        )
        fname = f"patch_{i:03d}_brain{brain_id}.png"
        out_path = os.path.join(args.output_dir, fname)
        save_mips(patch, brain_id, voxel, out_path)
        print(f"  [{i + 1:>3}/{args.n_patches}] brain={brain_id}  voxel={voxel}")

    print(f"\nMIPs written to {args.output_dir}")


if __name__ == "__main__":
    main()
