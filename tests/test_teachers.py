"""Tests for configurable BM4D teacher construction."""

from unittest.mock import patch

import numpy as np
import pytest

from aind_exaspim_image_compression.machine_learning.noise_models import (
    NoiseModel,
)
from aind_exaspim_image_compression.machine_learning.teachers import (
    build_teacher,
)
from aind_exaspim_image_compression.machine_learning.transforms import (
    AnscombeTransform,
)


def test_raw_bm4d_preserves_legacy_call_and_clips():
    """Raw mode passes count data and configured sigma directly to BM4D."""
    raw = np.array([-2.0, 10.0, 80.0], dtype=np.float32)
    with patch(
        "aind_exaspim_image_compression.machine_learning.teachers.bm4d",
        return_value=np.array([-1.0, 11.0, 90.0]),
    ) as denoise:
        teacher = build_teacher(raw, "raw_bm4d", 16, max_count=75)

    np.testing.assert_array_equal(teacher, [0.0, 11.0, 75.0])
    np.testing.assert_array_equal(denoise.call_args.args[0], raw)
    assert denoise.call_args.args[1] == 16
    assert teacher.dtype == np.float32


def test_gat_bm4d_uses_normalized_unit_noise_scale():
    """GAT mode stabilizes counts and supplies the normalized noise sigma."""
    raw = np.array([0.0, 100.0, 1000.0], dtype=np.float32)
    model = NoiseModel(
        offset=37,
        variance_slope=1.8,
        variance_intercept=400,
    )

    def identity(stabilized, sigma):
        assert sigma > 0
        assert sigma < 1
        return stabilized

    with patch(
        "aind_exaspim_image_compression.machine_learning.teachers.bm4d",
        side_effect=identity,
    ) as denoise:
        teacher = build_teacher(raw, "gat_bm4d", 999, model)

    # The denoising inverse intentionally differs from an algebraic round trip
    # by gain / 4, but an identity denoiser still returns count-scale values.
    np.testing.assert_allclose(teacher, raw + 1.8 / 4, atol=2e-3)
    assert denoise.call_args.args[1] != 999


def test_gat_sigma_multiplier_scales_bm4d_sigma():
    """GAT strength control multiplies the calibrated normalized sigma."""
    model = NoiseModel(0, 1.8, 400)
    with patch(
        "aind_exaspim_image_compression.machine_learning.teachers.bm4d",
        side_effect=lambda stabilized, sigma: stabilized,
    ) as denoise:
        build_teacher(
            np.ones(3),
            "gat_bm4d",
            24,
            model,
            gat_sigma_multiplier=1.5,
        )

    normalized_sigma = denoise.call_args.args[1]
    baseline = AnscombeTransform(gain=1.8, read_noise=20).unit_noise_std
    assert normalized_sigma == pytest.approx(baseline * 1.5)


@pytest.mark.parametrize("multiplier", [0, -1, np.inf, np.nan])
def test_gat_sigma_multiplier_must_be_positive_and_finite(multiplier):
    """Invalid GAT strength settings fail before BM4D is called."""
    model = NoiseModel(0, 1.8, 400)
    with pytest.raises(ValueError, match="gat_sigma_multiplier"):
        build_teacher(
            np.ones(3),
            "gat_bm4d",
            24,
            model,
            gat_sigma_multiplier=multiplier,
        )


@pytest.mark.parametrize(
    "noise_model",
    [
        None,
        {
            "offset": 0,
            "variance_slope": 0,
            "variance_intercept": 1,
        },
    ],
)
def test_gat_bm4d_rejects_missing_or_zero_gain(noise_model):
    """GAT mode fails clearly without a usable calibrated gain."""
    with pytest.raises(ValueError):
        build_teacher(np.ones(2), "gat_bm4d", 16, noise_model)


def test_unknown_teacher_mode_is_rejected():
    """Misspelled modes cannot silently select another teacher."""
    with pytest.raises(ValueError, match="unknown teacher mode"):
        build_teacher(np.ones(2), "other", 16)
