"""Tests for noise-model validation and patch-fit aggregation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from aind_exaspim_image_compression.machine_learning import (
    clipped_poisson_gaussian as estimator_module,
)
from aind_exaspim_image_compression.machine_learning.noise_models import (
    NoiseEstimate,
    NoiseModel,
    aggregate_noise_estimates,
    clipped_normal_moments,
    estimate_clipped_poisson_gaussian,
    load_noise_models,
    plot_fit,
    save_noise_models,
    validate_noise_model,
    validate_noise_models,
)


def make_estimate(a, b, counts=(10, 20, 30)):
    """Construct a compact normalized estimate for aggregation tests."""
    counts = np.asarray(counts, dtype=float)
    values = np.linspace(0.2, 0.8, len(counts))
    return NoiseEstimate(
        a=a,
        b=b,
        level_means=values,
        level_stds=np.full(len(counts), 0.1),
        level_counts=counts,
        model_stds=np.full(len(counts), 0.1),
        smooth_fraction=0.75,
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


class CountSpaceAggregationTest(unittest.TestCase):
    """Tests conversion and robust aggregation of normalized patch fits."""

    def test_conversion_and_weighted_medians(self):
        """Retained smooth samples weight both count-space coordinates."""
        estimates = [
            make_estimate(0.01, 0.001, counts=(1,)),
            make_estimate(0.02, 0.002, counts=(20,)),
            make_estimate(0.50, 0.100, counts=(1,)),
        ]
        model, diagnostics = aggregate_noise_estimates(
            estimates,
            offset=10,
            max_count=100,
            sampling_weight=4,
            saturation_margin=500,
        )
        self.assertEqual(model, NoiseModel(10, 2, 40, 4, 500))
        np.testing.assert_allclose(diagnostics["patch_slopes"], [1, 2, 50])
        np.testing.assert_allclose(
            diagnostics["patch_intercepts"], [20, 40, 1500]
        )
        np.testing.assert_allclose(diagnostics["patch_weights"], [1, 20, 1])
        self.assertEqual(diagnostics["valid_fits"], 3)
        self.assertEqual(diagnostics["failed_fits"], 0)
        self.assertEqual(diagnostics["required_valid_fits"], 3)
        self.assertEqual(diagnostics["aggregate_slope"], 2)
        self.assertEqual(diagnostics["aggregate_intercept"], 40)

    def test_nonfinite_fits_are_rejected(self):
        """Unusable parameters and level counts do not contribute."""
        estimates = [
            make_estimate(0.01, 0.001),
            make_estimate(0.02, 0.002),
            make_estimate(0.03, 0.003),
            make_estimate(np.nan, 0.001),
            make_estimate(0.01, 0.001, counts=(0,)),
        ]
        _, diagnostics = aggregate_noise_estimates(
            estimates, offset=0, max_count=100, requested_patches=5
        )
        np.testing.assert_array_equal(diagnostics["valid_indices"], [0, 1, 2])
        np.testing.assert_array_equal(diagnostics["invalid_indices"], [3, 4])
        self.assertEqual(diagnostics["failed_fits"], 2)

    def test_nonfinite_count_space_conversion_is_rejected(self):
        """Finite normalized fits that overflow count space are discarded."""
        estimates = [make_estimate(0.01, 0.001)] * 3
        estimates.append(make_estimate(1e308, 0.001))
        with np.errstate(over="ignore"):
            _, diagnostics = aggregate_noise_estimates(
                estimates, offset=0, max_count=100, requested_patches=4
            )
        np.testing.assert_array_equal(diagnostics["invalid_indices"], [3])

    def test_insufficient_valid_patch_majority_fails(self):
        """Fewer than the required valid patch fits fails clearly."""
        estimates = [make_estimate(0.01, 0.001)] * 2 + [None] * 4
        with self.assertRaisesRegex(ValueError, "only 2 of 6.*at least 3"):
            aggregate_noise_estimates(
                estimates, offset=0, max_count=100, requested_patches=6
            )
        with self.assertRaisesRegex(ValueError, "at least 3"):
            aggregate_noise_estimates(
                [make_estimate(0.01, 0.001)] * 2,
                offset=0,
                max_count=100,
            )

    def test_negative_final_intercept_is_not_clamped(self):
        """An unphysical aggregate intercept fails instead of becoming zero."""
        estimates = [make_estimate(0.01, -0.01)] * 3
        with self.assertRaisesRegex(ValueError, "intercept is negative"):
            aggregate_noise_estimates(estimates, offset=0, max_count=100)

    def test_invalid_aggregation_configuration(self):
        """Invalid white levels and requested patch counts are rejected."""
        estimates = [make_estimate(0.01, 0.001)] * 3
        for max_count in (0, np.inf, "bad"):
            with (
                self.subTest(max_count=max_count),
                self.assertRaises(ValueError),
            ):
                aggregate_noise_estimates(
                    estimates, offset=0, max_count=max_count
                )
        for requested in (0, 1.5, True, np.inf, "bad"):
            with (
                self.subTest(requested=requested),
                self.assertRaises(ValueError),
            ):
                aggregate_noise_estimates(
                    estimates,
                    offset=0,
                    max_count=100,
                    requested_patches=requested,
                )
        with self.assertRaisesRegex(ValueError, "cannot outnumber"):
            aggregate_noise_estimates(
                estimates, offset=0, max_count=100, requested_patches=2
            )


class EstimatorInterfaceTest(unittest.TestCase):
    """Tests deterministic public estimator behavior and diagnostics."""

    def test_clipped_normal_moments(self):
        """Closed-form clipped moments match fixed reference values."""
        mean, std = clipped_normal_moments(
            np.array([0.0, 0.5, -0.2, 1.2]),
            np.array([1.0, 0.2, 0.1, 0.1]),
        )
        np.testing.assert_allclose(
            mean,
            [0.3156268098137464, 0.5, 0.000849070261683, 0.999150929738317],
            rtol=1e-12,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            std,
            [
                0.39800627181949505,
                0.19774326622697488,
                0.007547605370972,
                0.007547605370966,
            ],
            rtol=1e-12,
            atol=1e-14,
        )

    def test_invalid_dimensions_modes_and_level_sets(self):
        """Invalid geometry, modes, and flat data fail clearly."""
        for shape in ((10,), (2, 2, 2, 2)):
            with (
                self.subTest(shape=shape),
                self.assertRaisesRegex(ValueError, "2-D image or a 3-D"),
            ):
                estimate_clipped_poisson_gaussian(np.zeros(shape))
        with self.assertRaisesRegex(ValueError, "mode must be"):
            estimate_clipped_poisson_gaussian(
                np.zeros((16, 16)), mode="temporal"
            )
        with self.assertRaisesRegex(RuntimeError, "Too few usable"):
            estimate_clipped_poisson_gaussian(
                np.zeros((32, 32)), min_samples_per_level=10
            )
        texture = np.random.default_rng(2).normal(size=(32, 32))
        with self.assertRaisesRegex(RuntimeError, "Too few usable"):
            estimate_clipped_poisson_gaussian(
                texture, min_samples_per_level=100
            )

        ramp = np.linspace(0, 1, 64)[None, :] + np.zeros((64, 1))
        with self.assertRaisesRegex(RuntimeError, "Too few usable"):
            estimate_clipped_poisson_gaussian(
                ramp, n_levels=60, min_samples_per_level=200
            )
        with patch.object(
            estimator_module,
            "_unbias_std_factor",
            return_value=np.nan,
        ):
            with self.assertRaisesRegex(RuntimeError, "Too few usable"):
                estimate_clipped_poisson_gaussian(
                    ramp, n_levels=10, min_samples_per_level=20
                )

    def test_inverse_mean_and_boundary_initialization(self):
        """Bisection inversion and boundary-only initialization are stable."""
        targets = np.array([0.1, 0.5, 0.9])
        intensities = estimator_module._invert_clipped_mean(
            targets, 0.01, 0.001
        )
        sigma = np.sqrt(np.maximum(0.01 * intensities + 0.001, 0))
        recovered, _ = clipped_normal_moments(intensities, sigma)
        np.testing.assert_allclose(recovered, targets, atol=1e-12)

        rng = np.random.default_rng(1)
        boundary = np.clip(rng.normal(0, 0.01, (64, 64)), 0, 1)
        with patch.object(
            estimator_module.np.linalg,
            "lstsq",
            wraps=estimator_module.np.linalg.lstsq,
        ) as least_squares:
            estimate = estimate_clipped_poisson_gaussian(
                boundary, n_levels=10, min_samples_per_level=20
            )
        self.assertEqual(least_squares.call_count, 0)
        self.assertTrue(np.isfinite(estimate.a))
        self.assertTrue(np.isfinite(estimate.b))

    def test_noise_estimate_curves_and_plot(self):
        """The result container evaluates and plots both fitted curves."""
        import matplotlib.pyplot as plt

        estimate = make_estimate(0.01, 0.001)
        sigma = estimate.sigma(np.array([-1.0, 0.0, 1.0]))
        np.testing.assert_allclose(
            sigma, [0.0, np.sqrt(0.001), np.sqrt(0.011)]
        )
        means, stds = estimate.clipped_curve(np.array([0.2, 0.8]))
        self.assertEqual(means.shape, (2,))
        self.assertEqual(stds.shape, (2,))

        axis = plot_fit(estimate, title="diagnostic")
        self.assertEqual(axis.get_title(), "diagnostic")
        self.assertEqual(len(axis.lines), 2)
        self.assertEqual(len(axis.collections), 1)
        plt.close(axis.figure)


if __name__ == "__main__":
    unittest.main()
