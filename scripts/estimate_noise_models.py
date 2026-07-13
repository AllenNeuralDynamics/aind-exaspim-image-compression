"""Estimate per-brain clipped Poisson--Gaussian noise models.

Raw level-0 sensor patches are normalized by the sensor white level without
subtracting the background offset. Each disjoint 3-D patch is fitted
independently with the clipped estimator, and valid fits are combined in
offset-subtracted count space using retained-sample-weighted medians.

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
    aggregate_noise_estimates,
    estimate_clipped_poisson_gaussian,
    plot_fit,
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


def _patch_bounds(center, patch_shape):
    """Return inclusive-exclusive spatial bounds for a centered patch."""
    center = np.asarray(center, dtype=np.int64)
    patch_shape = np.asarray(patch_shape, dtype=np.int64)
    start = center - patch_shape // 2
    return start, start + patch_shape


def _overlaps_any(bounds, accepted_bounds):
    """Return whether bounds overlap any previously accepted 3-D patch."""
    start, stop = bounds
    return any(
        np.all(start < other_stop) and np.all(other_start < stop)
        for other_start, other_stop in accepted_bounds
    )


def iter_tissue_patches(
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
    """Yield disjoint tissue-containing patches without retaining them.

    A candidate is accepted when the requested fraction of its voxels exceeds
    ``offset + min_signal_above_offset``. Each yielded tuple contains the
    patch, its center, and the cumulative candidate-location attempt count.
    Previously accepted patch bounds are excluded before image data are read.
    """
    if int(n_patches) < 1:
        raise ValueError("n_patches must be positive")
    if not 0 <= min_signal_fraction <= 1:
        raise ValueError("min_signal_fraction must be in [0, 1]")
    if max_attempt_factor < 1:
        raise ValueError("max_attempt_factor must be >= 1")
    threshold = float(offset) + float(min_signal_above_offset)
    max_attempts = int(np.ceil(n_patches * max_attempt_factor))
    accepted_bounds = []
    accepted = 0
    attempts = 0
    while accepted < n_patches and attempts < max_attempts:
        center = sample_patch_centers(
            image.shape[2:],
            patch_shape,
            1,
            rng,
            center_fraction=center_fraction,
        )[0]
        bounds = _patch_bounds(center, patch_shape)
        attempts += 1
        if _overlaps_any(bounds, accepted_bounds):
            continue
        patch = read_patch(image, center, patch_shape)
        if np.mean(patch > threshold) < min_signal_fraction:
            continue
        accepted_bounds.append(bounds)
        accepted += 1
        yield patch, center, attempts
    if accepted < n_patches:
        raise ValueError(
            f"found only {accepted} disjoint tissue-containing patches after "
            f"{attempts} attempts; lower min_patch_signal_fraction or "
            "min_signal_above_offset, increase max_patch_attempt_factor, or "
            "increase center_fraction"
        )


def make_diagnostic_plot(
    brain_id,
    estimates,
    centers,
    model,
    diagnostics,
    failure_counts,
    output_path,
):
    """Save the strongest reference fit and count-space fit distributions."""
    valid_indices = diagnostics["valid_indices"]
    best_valid_position = int(np.argmax(diagnostics["patch_weights"]))
    best_index = int(valid_indices[best_valid_position])
    best_estimate = estimates[best_index]

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fit_axis = axes[0, 0]
    slope_axis = axes[0, 1]
    intercept_axis = axes[1, 0]
    failure_axis = axes[1, 1]

    plot_fit(
        best_estimate,
        ax=fit_axis,
        title=(
            f"Highest-weight patch {best_index + 1} at {centers[best_index]} "
            f"({diagnostics['patch_weights'][best_valid_position]:.0f} "
            "smooth samples)"
        ),
    )

    patch_slopes = diagnostics["patch_slopes"]
    slope_axis.hist(
        patch_slopes,
        bins=min(12, max(1, len(patch_slopes))),
        color="tab:blue",
        alpha=0.75,
    )
    slope_axis.axvline(
        model.variance_slope,
        color="tab:red",
        linewidth=2,
        label=f"weighted median = {model.variance_slope:.4g}",
    )
    slope_axis.set(xlabel="A (count-space slope)", ylabel="Patch fits")
    slope_axis.legend(fontsize=8)

    patch_intercepts = diagnostics["patch_intercepts"]
    intercept_axis.hist(
        patch_intercepts,
        bins=min(12, max(1, len(patch_intercepts))),
        color="tab:purple",
        alpha=0.75,
    )
    intercept_axis.axvline(
        model.variance_intercept,
        color="tab:red",
        linewidth=2,
        label=f"weighted median = {model.variance_intercept:.4g}",
    )
    intercept_axis.set(xlabel="C (count-space intercept)", ylabel="Patch fits")
    intercept_axis.legend(fontsize=8)

    outcome_counts = Counter(failure_counts)
    outcome_counts["successful"] = diagnostics["valid_fits"]
    unclassified = diagnostics["failed_fits"] - sum(failure_counts.values())
    if unclassified > 0:
        outcome_counts["invalid estimate"] += unclassified
    labels = list(outcome_counts)
    failure_axis.barh(
        labels,
        [outcome_counts[label] for label in labels],
        color=[
            "tab:green" if label == "successful" else "tab:red"
            for label in labels
        ],
        alpha=0.75,
    )
    failure_axis.set(xlabel="Patches", title="Independent fit outcomes")

    failures = ", ".join(
        f"{label}: {count}" for label, count in sorted(failure_counts.items())
    )
    failures = failures or "none"
    figure.suptitle(
        f"Brain {brain_id}: Var = {model.variance_slope:.4g} X + "
        f"{model.variance_intercept:.4g}; valid fits "
        f"{diagnostics['valid_fits']}/{diagnostics['requested_patches']}; "
        f"fit failures: {failures}"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def estimate_brain(brain_id, args, rng, offsets, weights):
    """Stream patch fits for one brain and write its diagnostic plot."""
    if not np.isfinite(args.max_count) or args.max_count <= 0:
        raise ValueError("max_count must be finite and positive")
    prefix = get_img_prefix(brain_id, args.img_prefixes)
    image = img_util.read(prefix + str(args.level))
    offset = offsets.get(brain_id, 0.0)
    estimates = []
    centers = []
    failure_counts = Counter()
    attempts = 0
    patches = iter_tissue_patches(
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
    for patch, center, attempts in patches:
        centers.append(center)
        try:
            estimate = estimate_clipped_poisson_gaussian(
                patch / args.max_count,
                n_levels=args.n_levels,
                min_samples_per_level=args.min_samples_per_level,
                edge_tau=args.edge_tau,
                fit_loss=args.fit_loss,
                mode=args.mode,
            )
        except Exception as error:
            failure_counts[type(error).__name__] += 1
            estimate = None
        estimates.append(estimate)

    try:
        model, diagnostics = aggregate_noise_estimates(
            estimates,
            offset=offset,
            max_count=args.max_count,
            sampling_weight=weights.get(brain_id, 1.0),
            saturation_margin=args.saturation_margin,
            requested_patches=args.patches_per_brain,
        )
    except ValueError as error:
        failures = ", ".join(
            f"{label}={count}"
            for label, count in sorted(failure_counts.items())
        )
        detail = f"; estimator failures: {failures}" if failures else ""
        raise ValueError(f"brain {brain_id}: {error}{detail}") from error

    invalid_returned = sum(
        estimates[index] is not None
        for index in diagnostics["invalid_indices"]
    )
    if invalid_returned:
        failure_counts["nonfinite estimate"] += invalid_returned
    diagnostics["patches_attempted"] = attempts
    diagnostics["patches_accepted"] = len(centers)
    diagnostics["failure_counts"] = dict(failure_counts)
    make_diagnostic_plot(
        brain_id,
        estimates,
        centers,
        model,
        diagnostics,
        failure_counts,
        Path(args.diagnostics_dir) / f"{brain_id}.png",
    )
    return model, diagnostics


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
            "axis (default: 0.5)."
        ),
    )
    parser.add_argument("--patches-per-brain", type=int, default=512)
    parser.add_argument(
        "--patch-shape", type=int, nargs=3, default=(64, 64, 64)
    )
    parser.add_argument("--max-count", type=float, default=65535.0)
    parser.add_argument("--saturation-margin", type=int, default=500)
    parser.add_argument(
        "--mode",
        choices=("full", "slicewise"),
        default="full",
        help="Reference filtering mode (default: full).",
    )
    parser.add_argument(
        "--n-levels",
        "--level-count",
        dest="n_levels",
        type=int,
        default=60,
        help="Number of reference intensity level sets (default: 60).",
    )
    parser.add_argument(
        "--min-samples-per-level",
        "--min-level-samples",
        dest="min_samples_per_level",
        type=int,
        default=200,
        help="Minimum retained samples per level set (default: 200).",
    )
    parser.add_argument(
        "--edge-tau",
        "--edge-threshold",
        dest="edge_tau",
        type=float,
        default=2.0,
        help="Reference smooth-mask edge threshold (default: 2).",
    )
    parser.add_argument(
        "--fit-loss",
        choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
        default="soft_l1",
        help="Robust loss passed to SciPy least_squares (default: soft_l1).",
    )
    parser.add_argument(
        "--min-signal-above-offset",
        type=float,
        default=0,
        help="Patch occupancy threshold above the offset (default: 1 count).",
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
        help="Maximum candidate locations per requested patch (default: 20).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    """Run clipped noise calibration for all configured brains."""
    args = build_parser().parse_args(argv)
    brain_ids = util.read_txt(args.brain_ids)
    offsets = util.read_json(args.offsets) if args.offsets else {}
    weights = (
        util.read_json(args.sampling_weights) if args.sampling_weights else {}
    )
    rng = np.random.default_rng(args.seed)
    models = {}
    for brain_id in tqdm(brain_ids, desc="Estimate noise models"):
        model, diagnostics = estimate_brain(
            brain_id, args, rng, offsets, weights
        )
        models[brain_id] = model
        print(
            f"{brain_id}: A={model.variance_slope:.5g}, "
            f"C={model.variance_intercept:.5g}, "
            f"read_noise={model.read_noise:.3f}, "
            f"fits={diagnostics['valid_fits']}/"
            f"{diagnostics['requested_patches']}, "
            f"patches={diagnostics['patches_accepted']}/"
            f"{diagnostics['patches_attempted']} ({args.mode})"
        )
    save_noise_models(args.output, models)
    print(f"Wrote {len(models)} models to {args.output}")


if __name__ == "__main__":
    main()
