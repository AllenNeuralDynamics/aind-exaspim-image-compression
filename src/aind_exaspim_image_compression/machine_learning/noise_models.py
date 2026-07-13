"""Per-brain Poisson--Gaussian noise models and calibration helpers.

The downstream variance law is ``Var(Y | X) = A X + C`` for sensor counts
after subtracting a calibrated offset. ``NoiseModel`` stores the fitted slope
and intercept together with sampling metadata used by training and inference.
Patch calibration uses the clipped estimator re-exported from this module.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from aind_exaspim_image_compression.machine_learning import (
    clipped_poisson_gaussian as _clipped,
)

NoiseEstimate = _clipped.NoiseEstimate
clipped_normal_moments = _clipped.clipped_normal_moments
estimate_clipped_poisson_gaussian = _clipped.estimate_clipped_poisson_gaussian
plot_fit = _clipped.plot_fit


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


def _weighted_median(values, weights):
    """Return the first value whose cumulative weight reaches one half."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    location = np.searchsorted(
        np.cumsum(sorted_weights), np.sum(sorted_weights) / 2.0, side="left"
    )
    return float(sorted_values[min(location, len(sorted_values) - 1)])


def _valid_estimate_weight(estimate):
    """Return retained smooth samples, or None for an invalid estimate."""
    try:
        parameters = np.asarray([estimate.a, estimate.b], dtype=float)
        counts = np.asarray(estimate.level_counts, dtype=float)
    except (AttributeError, TypeError, ValueError):
        return None
    if (
        not np.all(np.isfinite(parameters))
        or counts.ndim != 1
        or not counts.size
        or not np.all(np.isfinite(counts))
        or np.any(counts <= 0)
    ):
        return None
    weight = float(np.sum(counts))
    return weight if math.isfinite(weight) and weight > 0 else None


def _normalize_aggregation_inputs(estimates, max_count, requested_patches):
    """Validate and normalize aggregation sizing inputs."""
    estimates = list(estimates)
    try:
        max_count = float(max_count)
    except (TypeError, ValueError) as error:
        raise ValueError("max_count must be finite and positive") from error
    if not math.isfinite(max_count) or max_count <= 0:
        raise ValueError("max_count must be finite and positive")
    if requested_patches is None:
        requested_patches = len(estimates)
    if isinstance(requested_patches, (bool, np.bool_)):
        raise ValueError("requested_patches must be a positive integer")
    try:
        integer_requested_patches = int(requested_patches)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "requested_patches must be a positive integer"
        ) from error
    if (
        integer_requested_patches != requested_patches
        or integer_requested_patches < 1
    ):
        raise ValueError("requested_patches must be a positive integer")
    if len(estimates) > integer_requested_patches:
        raise ValueError("estimates cannot outnumber requested_patches")
    return estimates, max_count, integer_requested_patches


def _convert_valid_estimates(estimates, offset, max_count):
    """Return finite count-space parameters, weights, and source indices."""
    valid_indices = []
    patch_weights = []
    patch_slopes = []
    patch_intercepts = []
    for index, estimate in enumerate(estimates):
        weight = _valid_estimate_weight(estimate)
        if weight is None:
            continue
        slope = float(estimate.a) * max_count
        intercept = float(estimate.b) * max_count**2 + slope * float(offset)
        if not math.isfinite(slope) or not math.isfinite(intercept):
            continue
        valid_indices.append(index)
        patch_weights.append(weight)
        patch_slopes.append(slope)
        patch_intercepts.append(intercept)
    return (
        valid_indices,
        np.asarray(patch_weights, dtype=np.float64),
        np.asarray(patch_slopes, dtype=np.float64),
        np.asarray(patch_intercepts, dtype=np.float64),
    )


def aggregate_noise_estimates(
    estimates,
    offset,
    max_count=65535.0,
    sampling_weight=1.0,
    saturation_margin=500,
    requested_patches=None,
):
    """Convert normalized patch fits and aggregate them in count space.

    Each normalized estimate ``(a, b)`` is translated to the downstream law
    for offset-subtracted sensor counts using ``A = a * max_count`` and
    ``C = b * max_count**2 + A * offset``. Coordinate-wise weighted medians
    use each patch's total retained level-set sample count as its weight.

    At least ``max(3, ceil(requested_patches / 2))`` finite estimates are
    required. Individual negative intercepts are retained, but a negative
    aggregate count-space intercept is rejected rather than clamped.

    Returns
    -------
    tuple[NoiseModel, dict]
        The schema-compatible model and per-patch aggregation diagnostics.
    """
    estimates, max_count, requested_patches = _normalize_aggregation_inputs(
        estimates, max_count, requested_patches
    )
    converted = _convert_valid_estimates(estimates, offset, max_count)
    valid_indices, patch_weights, patch_slopes, patch_intercepts = converted

    required = max(3, math.ceil(requested_patches / 2))
    if len(valid_indices) < required:
        raise ValueError(
            f"only {len(valid_indices)} of {requested_patches} patch fits "
            f"were valid; at least {required} are required"
        )

    slope = _weighted_median(patch_slopes, patch_weights)
    intercept = _weighted_median(patch_intercepts, patch_weights)
    if intercept < 0:
        raise ValueError(
            "aggregate variance intercept is negative "
            f"({intercept:.6g} count^2)"
        )

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
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    all_indices = np.arange(len(estimates), dtype=np.int64)
    diagnostics = {
        "requested_patches": requested_patches,
        "required_valid_fits": required,
        "valid_fits": len(valid_indices),
        "failed_fits": requested_patches - len(valid_indices),
        "valid_indices": valid_indices,
        "invalid_indices": np.setdiff1d(all_indices, valid_indices),
        "patch_slopes": patch_slopes,
        "patch_intercepts": patch_intercepts,
        "patch_weights": patch_weights,
        "aggregate_slope": slope,
        "aggregate_intercept": intercept,
    }
    return model, diagnostics
