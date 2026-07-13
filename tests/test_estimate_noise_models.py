"""Tests for streaming tissue-aware noise-model calibration."""

import runpy
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from aind_exaspim_image_compression.machine_learning.noise_models import (
    NoiseEstimate,
    aggregate_noise_estimates,
)


def make_estimate(a=0.01, b=0.001, weight=60):
    """Return a compact valid estimator result for CLI tests."""
    return NoiseEstimate(
        a=a,
        b=b,
        level_means=np.array([0.2, 0.5, 0.8]),
        level_stds=np.array([0.1, 0.1, 0.1]),
        level_counts=np.array([weight / 3] * 3),
        model_stds=np.array([0.1, 0.1, 0.1]),
        smooth_fraction=0.75,
    )


class TissuePatchSamplingTest(unittest.TestCase):
    """Tests exclusion and streaming of calibration patch candidates."""

    @classmethod
    def setUpClass(cls):
        """Load the estimator script without invoking its CLI."""
        script = (
            Path(__file__).parents[1] / "scripts" / "estimate_noise_models.py"
        )
        cls.namespace = runpy.run_path(str(script))

    def test_rejects_exterior_before_yielding_tissue(self):
        """A near-zero candidate is skipped before tissue is streamed."""
        iterator = self.namespace["iter_tissue_patches"]
        sample_centers = MagicMock(return_value=[(2, 2, 2)])
        exterior = np.ones((2, 2, 2), dtype=np.float32)
        tissue = np.full((2, 2, 2), 10, dtype=np.float32)
        read_patch = MagicMock(side_effect=[exterior, tissue])
        image = MagicMock(shape=(1, 1, 4, 4, 4))

        with patch.dict(
            iterator.__globals__,
            {
                "sample_patch_centers": sample_centers,
                "read_patch": read_patch,
            },
        ):
            yielded = list(
                iterator(
                    image,
                    patch_shape=(2, 2, 2),
                    n_patches=1,
                    rng=np.random.default_rng(42),
                    offset=1,
                    min_signal_above_offset=4,
                    min_signal_fraction=0.05,
                    max_attempt_factor=2,
                )
            )

        self.assertEqual(len(yielded), 1)
        patch_value, center, attempts = yielded[0]
        self.assertEqual(attempts, 2)
        self.assertEqual(center, (2, 2, 2))
        np.testing.assert_array_equal(patch_value, tissue)

    def test_accepted_patches_are_disjoint(self):
        """Overlapping candidates are skipped before another image read."""
        iterator = self.namespace["iter_tissue_patches"]
        sample_centers = MagicMock(
            side_effect=[
                [(2, 2, 2)],
                [(2, 2, 2)],
                [(6, 6, 6)],
            ]
        )
        tissue = np.full((2, 2, 2), 10, dtype=np.float32)
        read_patch = MagicMock(return_value=tissue)
        image = MagicMock(shape=(1, 1, 8, 8, 8))
        with patch.dict(
            iterator.__globals__,
            {
                "sample_patch_centers": sample_centers,
                "read_patch": read_patch,
            },
        ):
            yielded = list(
                iterator(
                    image,
                    patch_shape=(2, 2, 2),
                    n_patches=2,
                    rng=np.random.default_rng(42),
                    offset=1,
                    min_signal_above_offset=1,
                    min_signal_fraction=0.05,
                    max_attempt_factor=2,
                )
            )
        self.assertEqual(
            [value[1] for value in yielded], [(2, 2, 2), (6, 6, 6)]
        )
        self.assertEqual(yielded[-1][2], 3)
        self.assertEqual(read_patch.call_count, 2)

    def test_center_fraction_bounds_patch_locations(self):
        """Every sampled patch fits fully inside the centered region."""
        sample_centers = self.namespace["sample_patch_centers"]
        centers = np.asarray(
            sample_centers(
                shape=(100, 100, 100),
                patch_shape=(10, 10, 10),
                n_patches=1000,
                rng=np.random.default_rng(42),
                center_fraction=0.5,
            )
        )
        self.assertTrue(np.all(centers >= 30))
        self.assertTrue(np.all(centers <= 70))

        for fraction in (0, 1.1, np.nan):
            with (
                self.subTest(fraction=fraction),
                self.assertRaises(ValueError),
            ):
                sample_centers(
                    (100, 100, 100),
                    (10, 10, 10),
                    1,
                    np.random.default_rng(42),
                    center_fraction=fraction,
                )
        with self.assertRaisesRegex(ValueError, "too small"):
            sample_centers(
                (100, 100, 100),
                (20, 20, 20),
                1,
                np.random.default_rng(42),
                center_fraction=0.1,
            )

    def test_sampling_budget_and_configuration_errors(self):
        """Impossible occupancy and invalid settings fail clearly."""
        iterator = self.namespace["iter_tissue_patches"]
        sample_centers = MagicMock(return_value=[(2, 2, 2)])
        read_patch = MagicMock(
            return_value=np.zeros((2, 2, 2), dtype=np.float32)
        )
        image = MagicMock(shape=(1, 1, 4, 4, 4))
        common = dict(
            image=image,
            patch_shape=(2, 2, 2),
            rng=np.random.default_rng(42),
            offset=1,
            min_signal_above_offset=4,
            min_signal_fraction=0.05,
            max_attempt_factor=2,
        )
        with patch.dict(
            iterator.__globals__,
            {
                "sample_patch_centers": sample_centers,
                "read_patch": read_patch,
            },
        ):
            with self.assertRaisesRegex(ValueError, "found only 0"):
                list(iterator(n_patches=1, **common))
        for changes in (
            {"n_patches": 0},
            {"n_patches": 1, "min_signal_fraction": 1.1},
            {"n_patches": 1, "max_attempt_factor": 0.5},
        ):
            settings = {**common, **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                list(iterator(**settings))


class NoiseEstimatorCliTest(unittest.TestCase):
    """Tests reference arguments and streamed estimator integration."""

    @classmethod
    def setUpClass(cls):
        """Load the estimator script without invoking its CLI."""
        script = (
            Path(__file__).parents[1] / "scripts" / "estimate_noise_models.py"
        )
        cls.namespace = runpy.run_path(str(script))

    def _args(self, diagnostics_dir):
        """Return a complete small calibration argument namespace."""
        return SimpleNamespace(
            img_prefixes="prefixes.json",
            level=0,
            patch_shape=(2, 2, 2),
            patches_per_brain=3,
            min_signal_above_offset=1,
            min_patch_signal_fraction=0.05,
            max_patch_attempt_factor=3,
            center_fraction=1.0,
            max_count=100.0,
            n_levels=10,
            min_samples_per_level=2,
            edge_tau=2.5,
            fit_loss="huber",
            mode="full",
            saturation_margin=500,
            diagnostics_dir=diagnostics_dir,
        )

    def test_reference_parser_defaults_and_options(self):
        """The CLI exposes reference controls and uses 32 full-3-D patches."""
        parser = self.namespace["build_parser"]()
        defaults = parser.parse_args([])
        self.assertEqual(defaults.patches_per_brain, 32)
        self.assertEqual(defaults.patch_shape, (128, 128, 128))
        self.assertEqual(defaults.mode, "full")
        self.assertEqual(defaults.n_levels, 60)
        self.assertEqual(defaults.min_samples_per_level, 200)
        self.assertEqual(defaults.edge_tau, 2.0)
        self.assertEqual(defaults.fit_loss, "soft_l1")
        self.assertEqual(defaults.saturation_margin, 500)
        self.assertEqual(defaults.min_signal_above_offset, 1)
        self.assertEqual(defaults.center_fraction, 0.5)

        custom = parser.parse_args(
            [
                "--mode",
                "slicewise",
                "--level-count",
                "24",
                "--min-level-samples",
                "50",
                "--edge-threshold",
                "3",
                "--fit-loss",
                "cauchy",
            ]
        )
        self.assertEqual(custom.mode, "slicewise")
        self.assertEqual(custom.n_levels, 24)
        self.assertEqual(custom.min_samples_per_level, 50)
        self.assertEqual(custom.edge_tau, 3)
        self.assertEqual(custom.fit_loss, "cauchy")

        obsolete = {
            "--window-shape",
            "--variance-estimator",
            "--gradient-quantile",
            "--structure-quantile",
            "--bins",
            "--lower-quantile",
        }
        self.assertTrue(obsolete.isdisjoint(parser._option_string_actions))

    def test_estimate_brain_streams_raw_normalized_patches(self):
        """Each raw patch is fitted before the generator advances."""
        estimate_brain = self.namespace["estimate_brain"]
        estimator = MagicMock(
            side_effect=[
                make_estimate(0.01, 0.001),
                make_estimate(0.01, 0.001),
                make_estimate(0.01, 0.001),
            ]
        )
        raw_patches = [
            np.full((2, 2, 2), value, dtype=np.float32)
            for value in (10, 50, 90)
        ]

        def stream(*args, **kwargs):
            """Require fitting to occur before yielding the next patch."""
            for index, raw_patch in enumerate(raw_patches):
                self.assertEqual(estimator.call_count, index)
                yield raw_patch, (index + 1,) * 3, index + 1

        diagnostic = MagicMock()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                estimate_brain.__globals__,
                {
                    "get_img_prefix": MagicMock(return_value="prefix/"),
                    "img_util": SimpleNamespace(
                        read=MagicMock(return_value=object())
                    ),
                    "iter_tissue_patches": stream,
                    "estimate_clipped_poisson_gaussian": estimator,
                    "make_diagnostic_plot": diagnostic,
                },
            ),
        ):
            model, diagnostics = estimate_brain(
                "brain",
                self._args(directory),
                np.random.default_rng(42),
                {"brain": 10},
                {"brain": 2},
            )

        self.assertEqual(model.offset, 10)
        self.assertEqual(model.variance_slope, 1)
        self.assertEqual(model.variance_intercept, 20)
        self.assertEqual(model.sampling_weight, 2)
        self.assertEqual(diagnostics["patches_attempted"], 3)
        self.assertEqual(diagnostics["patches_accepted"], 3)
        self.assertEqual(estimator.call_count, 3)
        for call, expected in zip(estimator.call_args_list, (0.1, 0.5, 0.9)):
            np.testing.assert_allclose(call.args[0], expected)
            self.assertEqual(call.kwargs["mode"], "full")
            self.assertEqual(call.kwargs["fit_loss"], "huber")
        diagnostic.assert_called_once()

    def test_estimator_failures_are_counted_and_fail_clearly(self):
        """Patch failures continue until aggregation enforces its threshold."""
        estimate_brain = self.namespace["estimate_brain"]

        def stream(*args, **kwargs):
            """Yield three accepted tissue patches."""
            for index in range(3):
                yield np.ones((2, 2, 2)), (index,) * 3, index + 1

        estimator = MagicMock(
            side_effect=[
                RuntimeError("flat"),
                make_estimate(),
                make_estimate(),
            ]
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                estimate_brain.__globals__,
                {
                    "get_img_prefix": MagicMock(return_value="prefix/"),
                    "img_util": SimpleNamespace(
                        read=MagicMock(return_value=object())
                    ),
                    "iter_tissue_patches": stream,
                    "estimate_clipped_poisson_gaussian": estimator,
                    "make_diagnostic_plot": MagicMock(),
                },
            ),
        ):
            with self.assertRaisesRegex(
                ValueError, "brain brain: only 2 of 3.*RuntimeError=1"
            ):
                estimate_brain(
                    "brain",
                    self._args(directory),
                    np.random.default_rng(42),
                    {"brain": 10},
                    {},
                )

    def test_aggregate_diagnostic_plot_is_written(self):
        """The diagnostic includes a reference fit and patch distributions."""
        make_plot = self.namespace["make_diagnostic_plot"]
        estimates = [
            make_estimate(0.01, 0.001, weight=30),
            make_estimate(0.02, 0.002, weight=90),
            make_estimate(0.03, 0.003, weight=60),
        ]
        model, diagnostics = aggregate_noise_estimates(
            estimates, offset=10, max_count=100
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "brain.png"
            make_plot(
                "brain",
                estimates,
                [(1, 1, 1), (2, 2, 2), (3, 3, 3)],
                model,
                diagnostics,
                Counter({"RuntimeError": 1}),
                output,
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
