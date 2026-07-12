"""Tests for tissue-aware noise-model patch sampling."""

from pathlib import Path
from unittest.mock import MagicMock

import runpy
import unittest

import numpy as np


class TissuePatchSamplingTest(unittest.TestCase):
    """Tests exclusion of near-black exterior candidate patches."""

    @classmethod
    def setUpClass(cls):
        """Load the estimator script without invoking its CLI."""
        script = (
            Path(__file__).parents[1] / "scripts" / "estimate_noise_models.py"
        )
        cls.namespace = runpy.run_path(str(script))

    def test_rejects_exterior_before_accepting_tissue(self):
        """A near-zero candidate is skipped and a tissue patch is retained."""
        sample = self.namespace["sample_tissue_patches"]
        sample.__globals__["sample_patch_centers"] = MagicMock(
            return_value=[(2, 2, 2)]
        )
        exterior = np.ones((2, 2, 2), dtype=np.float32)
        tissue = np.full((2, 2, 2), 10, dtype=np.float32)
        sample.__globals__["read_patch"] = MagicMock(
            side_effect=[exterior, tissue]
        )
        image = MagicMock(shape=(1, 1, 4, 4, 4))

        patches, centers, attempts = sample(
            image,
            patch_shape=(2, 2, 2),
            n_patches=1,
            rng=np.random.default_rng(42),
            offset=1,
            min_signal_above_offset=4,
            min_signal_fraction=0.05,
            max_attempt_factor=2,
        )

        self.assertEqual(attempts, 2)
        self.assertEqual(centers, [(2, 2, 2)])
        np.testing.assert_array_equal(patches[0], tissue)

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
        """Impossible occupancy and invalid sampling settings fail clearly."""
        sample = self.namespace["sample_tissue_patches"]
        sample.__globals__["sample_patch_centers"] = MagicMock(
            return_value=[(2, 2, 2)]
        )
        sample.__globals__["read_patch"] = MagicMock(
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
        with self.assertRaisesRegex(ValueError, "found only 0"):
            sample(n_patches=1, **common)
        for changes in (
            {"n_patches": 0},
            {"n_patches": 1, "min_signal_fraction": 1.1},
            {"n_patches": 1, "max_attempt_factor": 0.5},
        ):
            settings = {**common, **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                sample(**settings)


if __name__ == "__main__":
    unittest.main()
