"""Teacher construction for count-space denoising experiments."""

from collections.abc import Mapping

import numpy as np
from bm4d import bm4d

from aind_exaspim_image_compression.machine_learning.noise_models import (
    NoiseModel,
    validate_noise_model,
)
from aind_exaspim_image_compression.machine_learning.transforms import (
    AnscombeTransform,
)


TEACHER_MODES = ("raw_bm4d", "gat_bm4d")


def _coerce_noise_model(noise_model):
    """Return a validated noise model supplied as a model or mapping."""
    if isinstance(noise_model, NoiseModel):
        return validate_noise_model("teacher", noise_model)
    if isinstance(noise_model, Mapping):
        return validate_noise_model("teacher", noise_model)
    raise ValueError(
        "gat_bm4d requires a per-brain NoiseModel or noise-model mapping"
    )


def build_teacher(
    raw,
    mode,
    sigma_bm4d,
    noise_model=None,
    max_count=65535.0,
    gat_sigma_multiplier=1.0,
):
    """Build a clipped BM4D teacher in offset-subtracted count space.

    Parameters
    ----------
    raw : numpy.ndarray
        Offset-subtracted noisy counts.
    mode : {"raw_bm4d", "gat_bm4d"}
        Teacher algorithm. ``raw_bm4d`` preserves the legacy count-space
        behavior. ``gat_bm4d`` denoises after a generalized Anscombe transform.
    sigma_bm4d : float
        BM4D noise standard deviation for ``raw_bm4d``. It is ignored by the
        variance-stabilized mode, whose expected normalized noise scale is
        determined by the transform.
    noise_model : NoiseModel or Mapping, optional
        Per-brain Poisson--Gaussian calibration required by ``gat_bm4d``.
    max_count : float, optional
        Upper physical count bound after offset subtraction.
    gat_sigma_multiplier : float, optional
        Positive multiplier applied to the calibrated normalized noise
        standard deviation in ``gat_bm4d`` mode. Default is 1.

    Returns
    -------
    numpy.ndarray
        Floating-point teacher clipped to ``[0, max_count]``.
    """
    raw = np.asarray(raw, dtype=np.float32)
    max_count = float(max_count)
    if not np.isfinite(max_count) or max_count <= 0:
        raise ValueError("max_count must be finite and positive")

    if mode == "raw_bm4d":
        teacher = bm4d(raw, sigma_bm4d)
    elif mode == "gat_bm4d":
        gat_sigma_multiplier = float(gat_sigma_multiplier)
        if not np.isfinite(gat_sigma_multiplier) or gat_sigma_multiplier <= 0:
            raise ValueError("gat_sigma_multiplier must be finite and positive")
        model = _coerce_noise_model(noise_model)
        if model.variance_slope <= 0:
            raise ValueError(
                "gat_bm4d requires variance_slope > 0; the generalized "
                "Anscombe transform is undefined for zero gain"
            )
        gat = AnscombeTransform(
            gain=model.variance_slope,
            read_noise=model.read_noise,
            offset=0.0,
            max_count=max_count,
            unbiased_inverse=True,
        )
        stabilized = gat.forward(raw)
        teacher_stabilized = bm4d(
            stabilized, gat.unit_noise_std * gat_sigma_multiplier
        )
        teacher = gat.inverse_float(teacher_stabilized)
    else:
        choices = ", ".join(TEACHER_MODES)
        raise ValueError(f"unknown teacher mode {mode!r}; expected {choices}")

    return np.clip(teacher, 0.0, max_count).astype(np.float32)
