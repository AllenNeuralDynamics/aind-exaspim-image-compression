"""Tests for per-brain noise-model validation and estimation."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aind_exaspim_image_compression.machine_learning.noise_models import (
    NoiseModel,
    bin_lower_envelope,
    collect_window_statistics,
    estimate_noise_model,
    fit_variance_law,
    load_noise_models,
    reject_structured_windows,
    save_noise_models,
    validate_noise_model,
    validate_noise_models,
)


class NoiseModelSchemaTest(unittest.TestCase):
    """Tests strict schema loading and serialization."""

    def test_valid_model_and_defaults(self):
        """Required values load and optional metadata receives defaults."""
        model = validate_noise_model(
            "734348",
            {
                "offset": 37,
                "variance_slope": 1.8,
                "variance_intercept": 400,
            },
        )
        self.assertEqual(model, NoiseModel(37, 1.8, 400, 1.0, 0))
        self.assertEqual(model.read_noise, 20.0)

    def test_round_trip(self):
        """JSON serialization preserves every configured field."""
        model = NoiseModel(37, 1.8, 400, 4.0, 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "models.json"
            save_noise_models(path, {"734348": model})
            loaded = load_noise_models(path, required_brain_ids=[734348])
            self.assertEqual(loaded, {"734348": model})
            with open(path, "r") as file:
                raw = json.load(file)
            self.assertEqual(raw["734348"], model.to_dict())

    def test_invalid_values_fail_clearly(self):
        """Missing, unknown, non-finite, and unphysical values are rejected."""
        valid = {
            "offset": 0,
            "variance_slope": 1,
            "variance_intercept": 2,
        }
        invalid = [
            {"offset": 0},
            {**valid, "typo": 1},
            {**valid, "offset": np.nan},
            {**valid, "variance_slope": -1},
            {**valid, "variance_intercept": -1},
            {**valid, "sampling_weight": 0},
            {**valid, "saturation_margin": -1},
            {**valid, "saturation_margin": 1.5},
            {**valid, "offset": True},
            {**valid, "offset": object()},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_noise_model("brain", values)
        with self.assertRaises(ValueError):
            validate_noise_model("brain", [])
        with self.assertRaises(ValueError):
            validate_noise_models([])
        with self.assertRaisesRegex(ValueError, "missing brains: b"):
            validate_noise_models({"a": valid}, required_brain_ids=["b"])


class WindowStatisticsTest(unittest.TestCase):
    """Tests single-acquisition spatial-window measurements."""

    def test_excludes_padding_saturation_and_nonfinite(self):
        """Sensor-invalid windows receive explicit exclusion reasons."""
        good = np.arange(1, 65, dtype=np.float32).reshape(4, 4, 4) + 100
        zero = np.zeros_like(good)
        saturated = good.copy()
        saturated[0, 0, 0] = 65500
        nonfinite = good.copy()
        nonfinite[0, 0, 0] = np.nan
        stats = collect_window_statistics(
            [good, zero, saturated, nonfinite], window_shape=(4, 4, 4)
        )
        self.assertEqual(
            stats["exclusion"].tolist(),
            ["", "zero_padding", "near_saturation", "nonfinite"],
        )
        np.testing.assert_array_equal(
            stats["eligible"], [True, False, False, False]
        )

    def test_spatial_variance_and_invalid_shapes(self):
        """Variance is spatial and invalid window geometry is rejected."""
        first = np.arange(1, 65, dtype=np.float32).reshape(4, 4, 4) + 100
        stats = collect_window_statistics([first], window_shape=(4, 4, 4))
        expected = np.var(first, ddof=1, dtype=np.float64)
        self.assertAlmostEqual(stats["variance"][0], expected)
        with self.assertRaises(ValueError):
            collect_window_statistics([first[0]])
        with self.assertRaises(ValueError):
            collect_window_statistics([first], window_shape=(4, 4))
        with self.assertRaises(ValueError):
            collect_window_statistics([first], window_shape=(1, 4, 4))
        with self.assertRaises(ValueError):
            collect_window_statistics([first], zero_padding_fraction=0)

    def test_isolated_zero_is_not_treated_as_padding(self):
        """Clipped zeroes inside otherwise valid tissue remain eligible."""
        patch = np.full((4, 4, 4), 10, dtype=np.float32)
        patch[0, 0, 0] = 0
        stats = collect_window_statistics(
            [patch],
            window_shape=(4, 4, 4),
            zero_padding_fraction=0.5,
        )
        self.assertTrue(stats["eligible"][0])

    def test_neighbor_differences_suppress_smooth_structure(self):
        """First differences remove a linear ramp but preserve random noise."""
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 5, size=(16, 16, 16)).astype(np.float32)
        ramp = np.indices(noise.shape).sum(axis=0).astype(np.float32) * 20
        patch = 1000 + ramp + noise
        spatial = collect_window_statistics(
            [patch],
            window_shape=patch.shape,
            variance_estimator="spatial",
        )
        differences = collect_window_statistics(
            [patch],
            window_shape=patch.shape,
            variance_estimator="neighbor_difference",
        )
        self.assertGreater(spatial["variance"][0], 10000)
        self.assertAlmostEqual(differences["variance"][0], 25, delta=3)
        self.assertEqual(
            differences["variance_estimator"], "neighbor_difference"
        )
        with self.assertRaises(ValueError):
            collect_window_statistics([patch], variance_estimator="unknown")

    def test_structure_filter(self):
        """Empirical filters retain the quietest eligible windows."""
        stats = {
            "eligible": np.ones(4, dtype=bool),
            "exclusion": np.array(["", "", "", ""]),
            "mean": np.full(4, 10.0),
            "variance": np.array([1.0, 4.0, 100.0, 1.0]),
            "gradient_rms": np.array([1.0, 2.0, 100.0, 1.0]),
            "structure_fraction": np.array([0.1, 0.2, 0.1, 0.9]),
        }
        filtered = reject_structured_windows(
            stats, gradient_quantile=0.75, structure_quantile=0.75
        )
        np.testing.assert_array_equal(filtered["accepted"], [1, 1, 0, 0])
        self.assertEqual(filtered["exclusion"][2], "strong_gradient")
        self.assertEqual(filtered["exclusion"][3], "spatial_structure")
        for kwargs in (
            {"gradient_quantile": 0},
            {"structure_quantile": 1.1},
            {"offset": np.nan},
            {"min_signal_above_offset": -1},
            {"intensity_bins": 0},
        ):
            with self.assertRaises(ValueError):
                reject_structured_windows(stats, **kwargs)

        empty = dict(stats)
        empty["eligible"] = np.zeros(4, dtype=bool)
        filtered = reject_structured_windows(empty)
        self.assertFalse(np.any(filtered["accepted"]))
        self.assertEqual(filtered["gradient_limits"].size, 0)

    def test_rejects_exterior_without_bright_noise_bias(self):
        """Exterior is removed without bias against brighter noise."""
        stats = {
            "eligible": np.ones(4, dtype=bool),
            "exclusion": np.array(["", "", "", ""]),
            "mean": np.array([1.0, 7.0, 101.0, 1001.0]),
            "variance": np.array([1.0, 4.0, 100.0, 10000.0]),
            "gradient_rms": np.array([1.0, 2.0, 10.0, 100.0]),
            "structure_fraction": np.full(4, 0.1),
        }
        filtered = reject_structured_windows(
            stats,
            gradient_quantile=1.0,
            structure_quantile=1.0,
            offset=1.0,
            min_signal_above_offset=4.0,
        )
        np.testing.assert_array_equal(filtered["accepted"], [0, 1, 1, 1])
        np.testing.assert_allclose(filtered["gradient_score"], 1.0)
        self.assertEqual(filtered["exclusion"][0], "below_min_signal")


class VarianceFitTest(unittest.TestCase):
    """Tests lower-envelope binning and constrained robust fitting."""

    def test_recovers_variance_law_with_outlier(self):
        """Robust nonnegative fitting recovers a line despite one outlier."""
        means = np.linspace(0, 1000, 20)
        variances = 1.8 * means + 400
        variances[10] += 10000
        slope, intercept = fit_variance_law(means, variances)
        self.assertAlmostEqual(slope, 1.8, delta=0.05)
        self.assertAlmostEqual(intercept, 400, delta=30)

    def test_lower_envelope_and_end_to_end_estimate(self):
        """Model construction uses accepted, offset-corrected data."""
        means = np.repeat(np.linspace(100, 1100, 11), 10)
        variance = 2 * (means - 100) + 25
        variance += np.tile(np.linspace(0, 9, 10), 11)
        bin_x, bin_y, counts = bin_lower_envelope(
            means - 100,
            variance,
            n_bins=10,
            lower_quantile=0,
            min_bin_samples=5,
        )
        slope, intercept = fit_variance_law(bin_x, bin_y, counts)
        self.assertAlmostEqual(slope, 2, delta=0.05)
        self.assertAlmostEqual(intercept, 25, delta=20)
        stats = {
            "accepted": np.ones_like(means, dtype=bool),
            "mean": means,
            "variance": variance,
        }
        model, fit = estimate_noise_model(
            stats,
            offset=100,
            sampling_weight=4,
            saturation_margin=64,
            n_bins=10,
            lower_quantile=0,
            min_bin_samples=5,
        )
        self.assertAlmostEqual(model.variance_slope, 2, delta=0.05)
        self.assertEqual(model.sampling_weight, 4)
        self.assertEqual(model.saturation_margin, 64)
        self.assertEqual(len(fit["residuals"]), len(fit["bin_means"]))

    def test_invalid_fit_inputs(self):
        """Fit helpers reject underspecified and unphysical samples."""
        invalid_bin_kwargs = [
            ([1], [1], {}),
            ([1, 2], [1, 2], {"n_bins": 1}),
            ([1, 2], [1, 2], {"lower_quantile": -1}),
            ([1, 2], [1, 2], {"min_bin_samples": 0}),
            ([1, 1], [1, 2], {}),
            ([1, 2, 100], [1, 2, 3], {"min_bin_samples": 2}),
        ]
        for means, variances, kwargs in invalid_bin_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bin_lower_envelope(means, variances, **kwargs)

        invalid_fit_args = [
            ([1], [1], None),
            ([1, 2], [1], None),
            ([1, np.nan], [1, 2], None),
            ([-1, 2], [1, 2], None),
            ([1, 2], [1, 2], [1]),
            ([1, 2], [1, 2], [1, 0]),
        ]
        for means, variances, counts in invalid_fit_args:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                fit_variance_law(means, variances, counts)

        slope, intercept = fit_variance_law([0, 1], [2, 5])
        self.assertAlmostEqual(slope, 3)
        self.assertAlmostEqual(intercept, 2)


if __name__ == "__main__":
    unittest.main()
