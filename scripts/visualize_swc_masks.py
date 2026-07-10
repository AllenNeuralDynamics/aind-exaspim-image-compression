"""
Visualize SWC-only foreground masks over raw image patches.

The precompute cache stores only the final foreground mask (segmentation union
SWC), so the SWC contribution cannot be recovered from ``fg.npy``. This script
instead reads the cache's ``config.json``, loads the same raw images and SWCs,
centers patches on traced nodes, rebuilds the SWC-only mask with the configured
``skeleton_radius``, and writes Z/Y/X maximum-projection overlays.

No segmentation volumes or BM4D targets are loaded.

Examples
--------
    # Sample six traced locations using a training cache's configuration
    python scripts/visualize_swc_masks.py /results/patch_cache

    # Inspect one brain reproducibly
    python scripts/visualize_swc_masks.py /results/patch_cache/config.json
        --brain-id 123456 --n 10 --seed 7 --out swc_masks.png
"""

import argparse
from copy import deepcopy
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from aind_exaspim_dataset_utils.s3_util import get_img_prefix
from aind_exaspim_image_compression.machine_learning.data_handling import (
    TrainDataset,
)
from aind_exaspim_image_compression.utils import util


AXIS_TITLES = ("Z MIP", "Y MIP", "X MIP")


def read_config(path):
    """Reads ``config.json`` from a cache directory or explicit file path."""
    config_path = (
        os.path.join(path, "config.json") if os.path.isdir(path) else path
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Precompute config not found: {config_path}")
    return util.read_json(config_path)


def load_dataset(config, brain_id=None):
    """Loads raw images and SWCs using a saved precompute configuration.

    When no brain is requested explicitly, loading stops at the first brain
    with SWC nodes so the default preview stays quick.
    """
    brain_ids = util.read_txt(config["brain_ids_path"])
    if brain_id is not None:
        if brain_id not in brain_ids:
            raise ValueError(f"Brain ID not present in config: {brain_id}")
        brain_ids = [brain_id]

    swc_base = config.get("swc_pointers")
    if not swc_base:
        raise ValueError("Precompute config does not contain an SWC pointer")

    dataset = TrainDataset(
        tuple(config["patch_shape"]),
        skeleton_radius=config["skeleton_radius"],
    )
    for current_brain in tqdm(brain_ids, desc="Load images and SWCs"):
        img_path = (
            get_img_prefix(current_brain, config["img_prefixes_path"])
            + "0"
        )
        swc_pointer = deepcopy(swc_base)
        swc_pointer["path"] += f"/{current_brain}/world"
        dataset.ingest_brain(
            current_brain,
            img_path,
            segmentation_path=None,
            swc_pointer=swc_pointer,
        )
        if brain_id is None and current_brain in dataset.skeletons:
            print(f"Using first brain with SWCs: {current_brain}")
            break
    return dataset


def pick_examples(dataset, n, brain_id=None, seed=0):
    """Selects valid, full-patch centers from the loaded SWC nodes."""
    rng = np.random.default_rng(seed)
    candidates = list()
    brain_ids = [brain_id] if brain_id is not None else dataset.skeletons
    patch_shape = np.asarray(dataset.patch_shape)
    low_margin = patch_shape // 2
    high_margin = patch_shape - low_margin

    for current_brain in brain_ids:
        if current_brain not in dataset.skeletons:
            continue
        points = np.asarray(dataset.skeletons[current_brain])
        image_shape = np.asarray(dataset.imgs[current_brain].shape[-3:])
        valid = np.all(
            (points >= low_margin)
            & (points <= image_shape - high_margin),
            axis=1,
        )
        candidates.extend(
            (current_brain, tuple(point)) for point in points[valid]
        )

    if not candidates:
        raise ValueError("No in-bounds SWC nodes are available to visualize")
    count = min(int(n), len(candidates))
    indices = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[i] for i in indices]


def stretch(image, low, high):
    """Maps a count-space image to [0, 1] for display."""
    if high <= low:
        high = low + 1.0
    return np.clip((image.astype(np.float32) - low) / (high - low), 0, 1)


def overlay(gray, mask, alpha=0.5):
    """Overlays a boolean mask in red on a normalized grayscale image."""
    rgb = np.stack([gray, gray, gray], axis=-1)
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * red
    return rgb


def render(dataset, examples, output_path, low_pct=1.0, high_pct=99.9):
    """Renders three orthogonal SWC-mask overlays for each selected patch."""
    fig, axes = plt.subplots(
        len(examples),
        3,
        figsize=(9, 3 * len(examples)),
        squeeze=False,
    )
    for row, (brain_id, center) in enumerate(examples):
        raw = np.asarray(dataset.read_patch(brain_id, center), dtype=np.float32)
        mask = dataset.skeleton_mask(brain_id, center)
        low, high = np.percentile(raw, (low_pct, high_pct))
        for axis in range(3):
            raw_mip = raw.max(axis=axis)
            mask_mip = mask.max(axis=axis)
            axes[row, axis].imshow(
                overlay(stretch(raw_mip, low, high), mask_mip),
                interpolation="nearest",
            )
            axes[row, axis].set_xticks([])
            axes[row, axis].set_yticks([])
            if row == 0:
                axes[row, axis].set_title(AXIS_TITLES[axis])
        axes[row, 0].set_ylabel(
            f"{brain_id}\n{tuple(center)}\n{mask.sum()} voxels",
            fontsize=8,
        )

    fig.suptitle(
        f"SWC-only masks (radius={dataset.skeleton_radius} voxels)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        help="Precompute cache directory or path to its config.json.",
    )
    parser.add_argument("--brain-id", default=None)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="swc_mask_preview.png")
    parser.add_argument("--low-pct", type=float, default=1.0)
    parser.add_argument("--high-pct", type=float, default=99.9)
    args = parser.parse_args()

    config = read_config(args.config)
    dataset = load_dataset(config, brain_id=args.brain_id)
    examples = pick_examples(
        dataset,
        args.n,
        brain_id=args.brain_id,
        seed=args.seed,
    )
    render(
        dataset,
        examples,
        args.out,
        low_pct=args.low_pct,
        high_pct=args.high_pct,
    )


if __name__ == "__main__":
    main()
