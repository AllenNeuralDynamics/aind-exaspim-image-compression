"""Per-brain Poisson--Gaussian noise models and calibration helpers.

The variance law used by the denoising pipeline is ``Var(Y | X) = a X + c``.
``NoiseModel`` stores the fitted slope and intercept together with the raw
sensor offset and sampling metadata needed by later cache-generation steps.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import json
import math

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import nnls


@dataclass(frozen=True)
class NoiseModel:
    """Calibrated Poisson--Gaussian noise parameters for one brain.

    Parameters
    ----------
    offset : float
        Sensor pedestal subtracted before applying the variance law.
    variance_slope : float
        Nonnegative signal-dependent variance coefficient.
    variance_intercept : float
        Nonnegative signal-independent variance in squared count units.
    sampling_weight : float, optional
        Positive relative brain-sampling weight. Default is 1.
    saturation_margin : int, optional
        Nonnegative number of counts below the sensor maximum that should be
        treated as nearly saturated. Default is 0.
    """

    offset: float
    variance_slope: float
    variance_intercept: float
    sampling_weight: float = 1.0
    saturation_margin: int = 0

    @property
    def read_noise(self):
        """Return the fitted signal-independent noise standard deviation."""
        return math.sqrt(self.variance_intercept)

    def to_dict(self):
        """Return a JSON-serializable representation of the model."""
        return asdict(self)


_MODEL_FIELDS = {
    "offset",
    "variance_slope",
    "variance_intercept",
    "sampling_weight",
    "saturation_margin",
}
_REQUIRED_FIELDS = {"offset", "variance_slope", "variance_intercept"}


def _finite_number(value, field, brain_id):
    """Convert a numeric field to float and reject booleans/non-finite data."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"noise model for brain {brain_id!r}: {field} must be numeric"
        )
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"noise model for brain {brain_id!r}: {field} must be numeric"
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"noise model for brain {brain_id!r}: {field} must be finite"
        )
    return value


def validate_noise_model(brain_id, values):
    """Validate and normalize one noise-model mapping.

    ``sampling_weight`` and ``saturation_margin`` may be omitted and default
    to 1 and 0, respectively. Unknown keys are rejected so misspelled
    calibration fields cannot silently change an experiment.

    Parameters
    ----------
    brain_id : str
        Brain identifier used in error messages.
    values : Mapping or NoiseModel
        Model values to validate.

    Returns
    -------
    NoiseModel
        An immutable, normalized model instance.
    """
    brain_id = str(brain_id)
    if isinstance(values, NoiseModel):
        values = values.to_dict()
    if not isinstance(values, Mapping):
        raise ValueError(
            f"noise model for brain {brain_id!r} must be an object"
        )
    missing = _REQUIRED_FIELDS - set(values)
    if missing:
        raise ValueError(
            f"noise model for brain {brain_id!r} is missing fields: "
            f"{', '.join(sorted(missing))}"
        )
    unknown = set(values) - _MODEL_FIELDS
    if unknown:
        raise ValueError(
            f"noise model for brain {brain_id!r} has unknown fields: "
            f"{', '.join(sorted(unknown))}"
        )

    offset = _finite_number(values["offset"], "offset", brain_id)
    slope = _finite_number(
        values["variance_slope"], "variance_slope", brain_id
    )
    intercept = _finite_number(
        values["variance_intercept"], "variance_intercept", brain_id
    )
    weight = _finite_number(
        values.get("sampling_weight", 1.0), "sampling_weight", brain_id
    )
    margin_value = _finite_number(
        values.get("saturation_margin", 0),
        "saturation_margin",
        brain_id,
    )

    if slope < 0:
        raise ValueError(
            f"noise model for brain {brain_id!r}: variance_slope must be >= 0"
        )
    if intercept < 0:
        raise ValueError(
            "noise model for brain "
            f"{brain_id!r}: variance_intercept must be >= 0"
        )
    if weight <= 0:
        raise ValueError(
            f"noise model for brain {brain_id!r}: sampling_weight must be > 0"
        )
    if margin_value < 0 or not margin_value.is_integer():
        raise ValueError(
            "noise model for brain "
            f"{brain_id!r}: saturation_margin must be a nonnegative integer"
        )

    return NoiseModel(
        offset=offset,
        variance_slope=slope,
        variance_intercept=intercept,
        sampling_weight=weight,
        saturation_margin=int(margin_value),
    )


def validate_noise_models(values, required_brain_ids=None):
    """Validate a mapping of brain IDs to noise models.

    Parameters
    ----------
    values : Mapping
        Brain-ID keyed model configuration.
    required_brain_ids : Iterable[str], optional
        When provided, fail if any listed brain is absent.

    Returns
    -------
    dict[str, NoiseModel]
        Validated models keyed by string brain ID.
    """
    if not isinstance(values, Mapping):
        raise ValueError("noise-model configuration must be a JSON object")
    models = {
        str(brain_id): validate_noise_model(brain_id, model)
        for brain_id, model in values.items()
    }
    required = {str(brain_id) for brain_id in (required_brain_ids or ())}
    missing = required - set(models)
    if missing:
        raise ValueError(
            "noise-model configuration is missing brains: "
            f"{', '.join(sorted(missing))}"
        )
    return models


def load_noise_models(path, required_brain_ids=None):
    """Load and validate per-brain noise models from JSON."""
    with open(path, "r") as file:
        values = json.load(file)
    return validate_noise_models(values, required_brain_ids)


def save_noise_models(path, models):
    """Validate and write per-brain noise models as deterministic JSON."""
    validated = validate_noise_models(models)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(
            {
                brain_id: model.to_dict()
                for brain_id, model in sorted(validated.items())
            },
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def _window_slices(shape, window_shape):
    """Yield complete non-overlapping window slices for a 3-D array."""
    if len(shape) != 3 or len(window_shape) != 3:
        raise ValueError("patches and window_shape must be three-dimensional")
    if any(int(size) < 2 for size in window_shape):
        raise ValueError("window_shape values must be at least 2")
    for z in range(0, shape[0] - window_shape[0] + 1, window_shape[0]):
        for y in range(0, shape[1] - window_shape[1] + 1, window_shape[1]):
            for x in range(0, shape[2] - window_shape[2] + 1, window_shape[2]):
                yield (
                    slice(z, z + window_shape[0]),
                    slice(y, y + window_shape[1]),
                    slice(x, x + window_shape[2]),
                )


def _window_exclusion(window, saturation_threshold, zero_padding_fraction):
    """Return an exclusion reason and finite flag for one spatial window."""
    finite = np.all(np.isfinite(window))
    if not finite:
        return "nonfinite", False
    if np.mean(window <= 0) >= zero_padding_fraction:
        return "zero_padding", True
    saturated = np.any(window >= saturation_threshold)
    return ("near_saturation" if saturated else ""), True


def _neighbor_difference_variance(window):
    """Estimate noise variance from adjacent-voxel differences.

    For independent noise, half the variance of a first difference equals the
    voxel noise variance. The median across spatial axes reduces inflation
    from strongly oriented anatomical structure.
    """
    axis_estimates = []
    for axis in range(window.ndim):
        differences = np.diff(window, axis=axis)
        axis_estimates.append(
            float(np.var(differences, ddof=1, dtype=np.float64) / 2.0)
        )
    return float(np.median(axis_estimates))


def _measure_window(window, variance_estimator):
    """Calculate local noise and spatial-structure statistics."""
    mean = float(np.mean(window, dtype=np.float64))
    spatial_variance = float(np.var(window, ddof=1, dtype=np.float64))
    if variance_estimator == "spatial":
        variance = spatial_variance
    else:
        variance = _neighbor_difference_variance(window)
    gradients = np.gradient(window)
    gradient_rms = float(
        np.sqrt(
            sum(
                np.mean(gradient * gradient, dtype=np.float64)
                for gradient in gradients
            )
        )
    )
    smooth = gaussian_filter(window, sigma=1.0)
    smooth_variance = float(np.var(smooth, dtype=np.float64))
    structure_fraction = smooth_variance / max(
        spatial_variance, np.finfo(np.float32).eps
    )
    return mean, variance, gradient_rms, structure_fraction


def collect_window_statistics(
    patches,
    window_shape=(8, 8, 8),
    max_count=65535.0,
    saturation_margin=64,
    zero_padding_fraction=0.5,
    variance_estimator="spatial",
):
    """Measure local mean, variance, gradient, and structure per window.

    This is a single-acquisition estimator: variance is measured spatially
    within each small window. Anatomical structure can only increase that
    variance, so callers must reject structured windows and fit a robust lower
    envelope rather than treating all local variance as acquisition noise.

    ``variance_estimator`` may be ``"spatial"`` for ordinary within-window
    variance or ``"neighbor_difference"`` for half the variance of adjacent
    voxel differences. ``zero_padding_fraction`` distinguishes padded exterior
    from isolated zeroes introduced by clipping during background subtraction.
    Returns a dictionary of equally sized NumPy arrays. ``eligible`` excludes
    padding, non-finite data, and near-saturation; gradient and structure
    filtering is deferred until all windows are available so empirical
    thresholds can be applied per brain.
    """
    if not 0 < zero_padding_fraction <= 1:
        raise ValueError("zero_padding_fraction must be in (0, 1]")
    if variance_estimator not in {"spatial", "neighbor_difference"}:
        raise ValueError(
            "variance_estimator must be 'spatial' or 'neighbor_difference'"
        )
    patches = list(patches)

    result = {
        "mean": [],
        "variance": [],
        "gradient_rms": [],
        "structure_fraction": [],
        "eligible": [],
        "exclusion": [],
    }
    saturation_threshold = float(max_count) - float(saturation_margin)
    for patch in patches:
        patch = np.asarray(patch, dtype=np.float32)
        if patch.ndim != 3:
            raise ValueError(
                "each calibration patch must be three-dimensional"
            )
        for slices in _window_slices(patch.shape, tuple(window_shape)):
            window = patch[slices]
            exclusion, finite = _window_exclusion(
                window, saturation_threshold, zero_padding_fraction
            )
            if finite:
                values = _measure_window(window, variance_estimator)
                mean, variance, gradient_rms, structure_fraction = values
            else:
                values = (np.nan,) * 4
                (
                    mean,
                    variance,
                    gradient_rms,
                    structure_fraction,
                ) = values

            result["mean"].append(mean)
            result["variance"].append(variance)
            result["gradient_rms"].append(gradient_rms)
            result["structure_fraction"].append(structure_fraction)
            result["eligible"].append(not exclusion)
            result["exclusion"].append(exclusion)

    for key in ("mean", "variance", "gradient_rms", "structure_fraction"):
        result[key] = np.asarray(result[key], dtype=np.float64)
    result["eligible"] = np.asarray(result["eligible"], dtype=bool)
    result["exclusion"] = np.asarray(result["exclusion"], dtype=str)
    result["variance_estimator"] = variance_estimator
    return result


def reject_structured_windows(
    statistics,
    gradient_quantile=0.5,
    structure_quantile=0.5,
    offset=0.0,
    min_signal_above_offset=0.0,
    intensity_bins=16,
):
    """Reject high-gradient or spatially structured eligible windows.

    The gradient is divided by the local standard deviation before computing
    its threshold. For unstructured noise this score is approximately
    intensity-independent, so bright Poisson-noisy windows are not rejected
    merely because their raw gradient is larger than background gradients.
    Windows whose mean does not exceed the calibrated offset by
    ``min_signal_above_offset`` are excluded as uninformative exterior signal.

    Quantile thresholds are learned separately within equal-population
    intensity strata. This prevents the numerous dim exterior/background
    windows from setting structure cutoffs that reject every brighter tissue
    window. Returns a copy of the statistics with ``accepted``, normalized
    ``gradient_score``, and updated ``exclusion`` arrays.
    """
    if not 0 < gradient_quantile <= 1:
        raise ValueError("gradient_quantile must be in (0, 1]")
    if not 0 < structure_quantile <= 1:
        raise ValueError("structure_quantile must be in (0, 1]")
    if not np.isfinite(offset):
        raise ValueError("offset must be finite")
    if not np.isfinite(min_signal_above_offset) or min_signal_above_offset < 0:
        raise ValueError("min_signal_above_offset must be finite and >= 0")
    if int(intensity_bins) < 1:
        raise ValueError("intensity_bins must be positive")
    output = dict(statistics)
    eligible = np.asarray(statistics["eligible"], dtype=bool)
    exclusions = np.asarray(statistics["exclusion"], dtype="<U32").copy()
    means = np.asarray(statistics["mean"], dtype=np.float64)
    variances = np.asarray(statistics["variance"], dtype=np.float64)
    gradients = np.asarray(statistics["gradient_rms"], dtype=np.float64)
    gradient_score = gradients / np.sqrt(
        np.maximum(variances, np.finfo(np.float64).eps)
    )
    below_signal = eligible & (
        means - float(offset) < float(min_signal_above_offset)
    )
    exclusions[below_signal] = "below_min_signal"
    accepted = eligible & ~below_signal
    if np.any(accepted):
        candidate_indices = np.flatnonzero(accepted)
        candidate_signal = means[candidate_indices] - float(offset)
        n_bins = min(int(intensity_bins), len(candidate_indices))
        filter_edges = np.quantile(
            candidate_signal, np.linspace(0, 1, n_bins + 1)
        )
        bin_indices = np.searchsorted(
            filter_edges[1:-1], candidate_signal, side="right"
        )
        edge = np.zeros_like(accepted)
        structured = np.zeros_like(accepted)
        gradient_limits = []
        structure_limits = []
        structure_scores = np.asarray(statistics["structure_fraction"])
        for bin_index in range(n_bins):
            members = candidate_indices[bin_indices == bin_index]
            if not len(members):
                gradient_limits.append(np.nan)
                structure_limits.append(np.nan)
                continue
            gradient_limit = np.quantile(
                gradient_score[members], gradient_quantile
            )
            structure_limit = np.quantile(
                structure_scores[members], structure_quantile
            )
            edge[members] = gradient_score[members] > gradient_limit
            structured[members] = ~edge[members] & (
                structure_scores[members] > structure_limit
            )
            gradient_limits.append(float(gradient_limit))
            structure_limits.append(float(structure_limit))
        exclusions[edge] = "strong_gradient"
        exclusions[structured] = "spatial_structure"
        accepted &= ~(edge | structured)
        output["filter_bin_edges"] = filter_edges
        output["gradient_limits"] = np.asarray(gradient_limits)
        output["structure_limits"] = np.asarray(structure_limits)
    else:
        output["filter_bin_edges"] = np.asarray([])
        output["gradient_limits"] = np.asarray([])
        output["structure_limits"] = np.asarray([])
    output["gradient_score"] = gradient_score
    output["accepted"] = accepted
    output["exclusion"] = exclusions
    return output


def bin_lower_envelope(
    means,
    variances,
    n_bins=32,
    lower_quantile=0.2,
    min_bin_samples=5,
):
    """Return a lower envelope from equal-population intensity bins.

    Quantile bins prevent a dense near-background population from consuming
    nearly every bin while sparse brighter tissue is discarded for failing
    ``min_bin_samples``.
    """
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    valid = np.isfinite(means) & np.isfinite(variances) & (variances >= 0)
    means, variances = means[valid], variances[valid]
    if means.size < 2:
        raise ValueError("at least two valid windows are required")
    if int(n_bins) < 2:
        raise ValueError("n_bins must be at least 2")
    if not 0 <= lower_quantile <= 1:
        raise ValueError("lower_quantile must be in [0, 1]")
    if int(min_bin_samples) < 1:
        raise ValueError("min_bin_samples must be positive")

    lo, hi = float(np.min(means)), float(np.max(means))
    if lo == hi:
        raise ValueError("window means do not span an intensity range")
    edges = np.quantile(means, np.linspace(0, 1, int(n_bins) + 1))
    indices = np.searchsorted(edges[1:-1], means, side="right")
    bin_means, bin_variances, bin_counts = [], [], []
    for index in range(int(n_bins)):
        selected = indices == index
        count = int(np.sum(selected))
        if count < int(min_bin_samples):
            continue
        bin_means.append(float(np.median(means[selected])))
        bin_variances.append(
            float(np.quantile(variances[selected], lower_quantile))
        )
        bin_counts.append(count)
    if len(bin_means) < 2:
        raise ValueError(
            "fewer than two populated intensity bins; sample more patches or "
            "reduce min_bin_samples"
        )
    return (
        np.asarray(bin_means, dtype=np.float64),
        np.asarray(bin_variances, dtype=np.float64),
        np.asarray(bin_counts, dtype=np.int64),
    )


def fit_variance_law(means, variances, counts=None, max_iterations=20):
    """Fit a nonnegative robust line to lower-envelope variance samples.

    The constrained fit uses iteratively reweighted least squares with Huber
    weights. Both the variance slope and intercept are constrained to be
    nonnegative by nonnegative least squares.
    """
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    if means.shape != variances.shape or means.ndim != 1 or means.size < 2:
        raise ValueError("means and variances must be equal 1-D arrays")
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(variances)):
        raise ValueError("fit inputs must be finite")
    if np.any(means < 0) or np.any(variances < 0):
        raise ValueError("fit inputs must be nonnegative")
    if counts is None:
        base_weights = np.ones_like(means)
    else:
        counts = np.asarray(counts, dtype=np.float64)
        if counts.shape != means.shape or np.any(counts <= 0):
            raise ValueError("counts must be positive and match means")
        base_weights = counts / np.max(counts)

    design = np.column_stack((means, np.ones_like(means)))
    robust_weights = np.ones_like(means)
    coefficients = np.zeros(2, dtype=np.float64)
    for _ in range(int(max_iterations)):
        weights = np.sqrt(base_weights * robust_weights)
        updated, _ = nnls(design * weights[:, None], variances * weights)
        residuals = variances - design @ updated
        center = np.median(residuals)
        scale = 1.4826 * np.median(np.abs(residuals - center))
        if scale <= np.finfo(np.float64).eps:
            coefficients = updated
            break
        standardized = np.abs(residuals - center) / (1.345 * scale)
        next_weights = np.ones_like(standardized)
        large = standardized > 1
        next_weights[large] = 1.0 / standardized[large]
        if np.allclose(updated, coefficients, rtol=1e-8, atol=1e-10):
            coefficients = updated
            break
        coefficients = updated
        robust_weights = next_weights
    return float(coefficients[0]), float(coefficients[1])


def estimate_noise_model(
    statistics,
    offset,
    sampling_weight=1.0,
    saturation_margin=64,
    n_bins=32,
    lower_quantile=0.2,
    min_bin_samples=5,
):
    """Fit and validate a ``NoiseModel`` from accepted window statistics."""
    accepted = np.asarray(statistics["accepted"], dtype=bool)
    means = np.maximum(np.asarray(statistics["mean"])[accepted] - offset, 0)
    variances = np.asarray(statistics["variance"])[accepted]
    bin_means, bin_variances, bin_counts = bin_lower_envelope(
        means,
        variances,
        n_bins=n_bins,
        lower_quantile=lower_quantile,
        min_bin_samples=min_bin_samples,
    )
    slope, intercept = fit_variance_law(bin_means, bin_variances, bin_counts)
    model = validate_noise_model(
        "calibration",
        {
            "offset": offset,
            "variance_slope": slope,
            "variance_intercept": intercept,
            "sampling_weight": sampling_weight,
            "saturation_margin": saturation_margin,
        },
    )
    fit = {
        "bin_means": bin_means,
        "bin_variances": bin_variances,
        "bin_counts": bin_counts,
        "residuals": bin_variances - (slope * bin_means + intercept),
    }
    return model, fit
