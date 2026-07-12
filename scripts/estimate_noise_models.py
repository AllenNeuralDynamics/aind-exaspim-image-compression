"""Estimate per-brain Poisson--Gaussian noise models from level-0 images.

This single-acquisition estimator samples raw-resolution patches, divides
them into small windows, rejects padding, near-saturation, strong gradients,
and spatially structured windows, then robustly fits a lower-envelope line to
local spatial variance versus offset-subtracted local mean. The lower envelope
is essential because residual anatomical structure inflates spatial variance.
For volumes that were background-subtracted and clipped before storage, the
fitted intercept is the effective residual variance in that stored volume; the
original camera read-noise variance is not identifiable from one clipped
acquisition.

Example
-------
python scripts/estimate_noise_models.py \
    --brain-ids /data/train_brain_ids.txt \
    --img-prefixes /data/exaspim_image_prefixes.json \
    --offsets /data/exaspim_background_offsets.json \
    --output /results/noise_models.json \
    --diagnostics-dir /results/noise_model_diagnostics
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
from aind_exaspim_dataset_utils.s3_util import get_img_prefix
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aind_exaspim_image_compression.machine_learning.noise_models import (  # noqa: E402,E501
    collect_window_statistics,
    estimate_noise_model,
    reject_structured_windows,
    save_noise_models,
)
from aind_exaspim_image_compression.utils import img_util, util  # noqa: E402


def sample_patch_centers(
    shape, patch_shape, n_patches, rng, center_fraction=1.0
):
    """Draw valid centers from a centered fraction of a 3-D volume.

    The complete patch, rather than only its center voxel, must fit inside the
    centered region. A fraction of 1 uses the full volume; 0.5 uses the central
    half of every spatial axis.
    """
    shape = np.asarray(shape, dtype=np.int64)
    patch_shape = np.asarray(patch_shape, dtype=np.int64)
    if shape.shape != (3,) or patch_shape.shape != (3,):
        raise ValueError("shape and patch_shape must have three values")
    if np.any(patch_shape <= 0) or np.any(patch_shape > shape):
        raise ValueError("patch_shape must be positive and fit in the image")
    if not np.isfinite(center_fraction) or not 0 < center_fraction <= 1:
        raise ValueError("center_fraction must be finite and in (0, 1]")

    region_size = np.floor(shape * float(center_fraction)).astype(np.int64)
    region_start = (shape - region_size) // 2
    region_stop = region_start + region_size
    low = region_start + patch_shape // 2
    high = region_stop - (patch_shape - patch_shape // 2) + 1
    if np.any(high <= low):
        raise ValueError(
            "center_fraction defines a region too small for patch_shape"
        )
    return [
        tuple(rng.integers(low, high, endpoint=False).tolist())
        for _ in range(int(n_patches))
    ]


def read_patch(image, center, patch_shape):
    """Read one spatial patch from an image in ``(t, c, z, y, x)`` order."""
    slices = img_util.get_slices(center, patch_shape)
    return np.asarray(image[(0, 0, *slices)], dtype=np.float32)


def sample_patches(image, centers, patch_shape):
    """Read calibration patches at a fixed sequence of centers."""
    return [read_patch(image, center, patch_shape) for center in centers]


def sample_tissue_patches(
    image,
    patch_shape,
    n_patches,
    rng,
    offset,
    min_signal_above_offset,
    min_signal_fraction,
    max_attempt_factor,
    center_fraction=1.0,
):
    """Rejection-sample patches containing a minimum amount of tissue signal.

    Exterior voxels in background-subtracted volumes may be 0 or 1 rather
    than exact padding. A candidate is retained only when the requested
    fraction of its voxels exceeds ``offset + min_signal_above_offset``.

    Returns
    -------
    tuple[list[numpy.ndarray], list[tuple[int]], int]
        Accepted patches, their centers, and total candidates read.
    """
    if int(n_patches) < 1:
        raise ValueError("n_patches must be positive")
    if not 0 <= min_signal_fraction <= 1:
        raise ValueError("min_signal_fraction must be in [0, 1]")
    if max_attempt_factor < 1:
        raise ValueError("max_attempt_factor must be >= 1")
    threshold = float(offset) + float(min_signal_above_offset)
    max_attempts = int(np.ceil(n_patches * max_attempt_factor))
    patches, centers = [], []
    attempts = 0
    while len(patches) < n_patches and attempts < max_attempts:
        center = sample_patch_centers(
            image.shape[2:],
            patch_shape,
            1,
            rng,
            center_fraction=center_fraction,
        )[0]
        patch = read_patch(image, center, patch_shape)
        attempts += 1
        if np.mean(patch > threshold) < min_signal_fraction:
            continue
        patches.append(patch)
        centers.append(center)
    if len(patches) < n_patches:
        raise ValueError(
            f"found only {len(patches)} tissue-containing patches after "
            f"{attempts} attempts; lower min_signal_fraction or "
            "min_signal_above_offset, or increase max_attempt_factor"
        )
    return patches, centers, attempts


def make_diagnostic_plot(
    brain_id,
    statistics,
    model,
    fit,
    output_path,
    max_count=65535.0,
):
    """Save samples, fitted curve, residuals, and exclusions for one brain."""
    accepted = statistics["accepted"]
    mean_counts = statistics["mean"] - model.offset
    excluded = ~accepted
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    axes[0, 0].scatter(
        mean_counts[excluded],
        statistics["variance"][excluded],
        s=5,
        alpha=0.2,
        color="0.65",
        label="excluded",
    )
    axes[0, 0].scatter(
        mean_counts[accepted],
        statistics["variance"][accepted],
        s=6,
        alpha=0.3,
        color="tab:blue",
        label="accepted",
    )
    axes[0, 0].scatter(
        fit["bin_means"],
        fit["bin_variances"],
        s=30,
        color="tab:orange",
        label="lower envelope",
    )
    x_max = max(float(np.max(fit["bin_means"])), 1.0)
    line_x = np.linspace(0, x_max, 200)
    axes[0, 0].plot(
        line_x,
        model.variance_slope * line_x + model.variance_intercept,
        color="tab:red",
        label="fit",
    )
    saturation_start = max_count - model.saturation_margin - model.offset
    if saturation_start <= x_max:
        axes[0, 0].axvspan(
            saturation_start,
            max_count - model.offset,
            color="tab:red",
            alpha=0.08,
            label="near saturation",
        )
    axes[0, 0].set(xlabel="Mean signal (counts)", ylabel="Variance")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].axhline(0, color="0.3", linewidth=1)
    axes[0, 1].scatter(
        fit["bin_means"], fit["residuals"], s=25, color="tab:purple"
    )
    axes[0, 1].set(xlabel="Mean signal (counts)", ylabel="Fit residual")
    axes[0, 1].set_title(
        f"Fit range: {fit['bin_means'].min():.1f}–"
        f"{fit['bin_means'].max():.1f} counts"
    )

    axes[1, 0].hist(fit["residuals"], bins=min(20, len(fit["residuals"])))
    axes[1, 0].set(xlabel="Fit residual", ylabel="Envelope bins")

    counts = Counter(statistics["exclusion"].tolist())
    counts["accepted"] = int(np.sum(accepted))
    counts.pop("", None)
    labels = list(counts)
    axes[1, 1].barh(labels, [counts[label] for label in labels])
    patch_summary = ""
    if "patches_attempted" in statistics:
        patch_summary = (
            f"; patches {statistics['patches_accepted']}/"
            f"{statistics['patches_attempted']}"
        )
    axes[1, 1].set(xlabel="Windows", title=f"Filtering summary{patch_summary}")

    estimator = statistics.get("variance_estimator", "spatial").replace(
        "_", " "
    )
    figure.suptitle(
        f"Brain {brain_id}: Var = {model.variance_slope:.4g} X + "
        f"{model.variance_intercept:.4g} ({estimator} fit)"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def estimate_brain(brain_id, args, rng, offsets, weights):
    """Estimate one brain model and write its diagnostic plot."""
    prefix = get_img_prefix(brain_id, args.img_prefixes)
    image = img_util.read(prefix + str(args.level))
    offset = offsets.get(brain_id, 0.0)
    patches, centers, attempts = sample_tissue_patches(
        image,
        args.patch_shape,
        args.patches_per_brain,
        rng,
        offset,
        args.min_signal_above_offset,
        args.min_patch_signal_fraction,
        args.max_patch_attempt_factor,
        center_fraction=args.center_fraction,
    )

    statistics = collect_window_statistics(
        patches,
        window_shape=args.window_shape,
        max_count=args.max_count,
        saturation_margin=args.saturation_margin,
        zero_padding_fraction=args.zero_padding_fraction,
        variance_estimator=args.variance_estimator,
    )
    statistics = reject_structured_windows(
        statistics,
        gradient_quantile=args.gradient_quantile,
        structure_quantile=args.structure_quantile,
        offset=offset,
        min_signal_above_offset=args.min_signal_above_offset,
        intensity_bins=args.filter_intensity_bins,
    )
    statistics["patches_attempted"] = attempts
    statistics["patches_accepted"] = len(centers)
    model, fit = estimate_noise_model(
        statistics,
        offset=offset,
        sampling_weight=weights.get(brain_id, 1.0),
        saturation_margin=args.saturation_margin,
        n_bins=args.bins,
        lower_quantile=args.lower_quantile,
        min_bin_samples=args.min_bin_samples,
    )
    statistics["fit_signal_min"] = float(np.min(fit["bin_means"]))
    statistics["fit_signal_max"] = float(np.max(fit["bin_means"]))
    make_diagnostic_plot(
        brain_id,
        statistics,
        model,
        fit,
        Path(args.diagnostics_dir) / f"{brain_id}.png",
        max_count=args.max_count,
    )
    return model, statistics


def build_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain-ids", default="/data/train_brain_ids.txt")
    parser.add_argument(
        "--img-prefixes", default="/data/exaspim_image_prefixes.json"
    )
    parser.add_argument(
        "--offsets", default="/data/exaspim_background_offsets.json"
    )
    parser.add_argument(
        "--sampling-weights",
        default=None,
        help="Optional JSON mapping brain IDs to positive relative weights.",
    )
    parser.add_argument("--output", default="/results/noise_models.json")
    parser.add_argument(
        "--diagnostics-dir", default="/results/noise_model_diagnostics"
    )
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument(
        "--center-fraction",
        type=float,
        default=0.5,
        help=(
            "Restrict patches to the centered fraction of every spatial "
            "axis; 0.5 uses the central half (default: 1.0)."
        ),
    )
    parser.add_argument("--patches-per-brain", type=int, default=512)
    parser.add_argument(
        "--patch-shape", type=int, nargs=3, default=(128, 128, 128)
    )
    parser.add_argument(
        "--window-shape", type=int, nargs=3, default=(32, 32, 32)
    )
    parser.add_argument("--max-count", type=float, default=65535.0)
    parser.add_argument("--saturation-margin", type=int, default=64)
    parser.add_argument(
        "--variance-estimator",
        choices=("spatial", "neighbor_difference"),
        default="spatial",
        help=(
            "Window variance estimator; neighbor_difference suppresses "
            "smooth local tissue structure (default: spatial)."
        ),
    )
    parser.add_argument(
        "--zero-padding-fraction",
        type=float,
        default=0.5,
        help=(
            "Reject a window as padding when at least this fraction is "
            "nonpositive (default: 0.5)."
        ),
    )
    parser.add_argument("--gradient-quantile", type=float, default=0.5)
    parser.add_argument("--structure-quantile", type=float, default=0.5)
    parser.add_argument(
        "--filter-intensity-bins",
        type=int,
        default=16,
        help=(
            "Equal-population intensity strata used for gradient and "
            "structure rejection (default: 16)."
        ),
    )
    parser.add_argument(
        "--min-signal-above-offset",
        type=float,
        default=0,
        help=(
            "Minimum window signal above the per-brain offset and patch "
            "occupancy threshold (default: 4 counts)."
        ),
    )
    parser.add_argument(
        "--min-patch-signal-fraction",
        type=float,
        default=0.05,
        help=(
            "Minimum fraction of patch voxels above the signal threshold "
            "(default: 0.05)."
        ),
    )
    parser.add_argument(
        "--max-patch-attempt-factor",
        type=float,
        default=20.0,
        help="Maximum candidate reads per requested patch (default: 20).",
    )
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--lower-quantile", type=float, default=0.2)
    parser.add_argument("--min-bin-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    """Run calibration for all configured brains."""
    args = build_parser().parse_args(argv)
    brain_ids = util.read_txt(args.brain_ids)
    offsets = util.read_json(args.offsets) if args.offsets else {}
    weights = (
        util.read_json(args.sampling_weights) if args.sampling_weights else {}
    )
    rng = np.random.default_rng(args.seed)
    models = {}
    for brain_id in tqdm(brain_ids, desc="Estimate noise models"):
        model, statistics = estimate_brain(
            brain_id, args, rng, offsets, weights
        )
        models[brain_id] = model
        print(
            f"{brain_id}: a={model.variance_slope:.5g}, "
            f"c={model.variance_intercept:.5g}, "
            f"read_noise={model.read_noise:.3f}, "
            f"windows={int(np.sum(statistics['accepted']))}, "
            f"patches={statistics['patches_accepted']}/"
            f"{statistics['patches_attempted']}, "
            f"fit_range={statistics['fit_signal_min']:.1f}–"
            f"{statistics['fit_signal_max']:.1f} counts "
            f"({statistics['variance_estimator']})"
        )
    save_noise_models(args.output, models)
    print(f"Wrote {len(models)} models to {args.output}")


if __name__ == "__main__":
    main()
