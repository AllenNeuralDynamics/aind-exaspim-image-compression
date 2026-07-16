"""Tests for the SWC-mask visualization script."""

from pathlib import Path

import runpy
import unittest

import numpy as np


class _Image:
    """Minimal image object exposing a shape."""

    shape = (1, 1, 10, 10, 10)


class _Dataset:
    """Minimal dataset for testing SWC center selection."""

    patch_shape = (4, 4, 4)
    imgs = {"brain": _Image()}
    skeletons = {
        "brain": np.array(
            [
                [1, 5, 5],  # outside the low z margin
                [2, 2, 2],  # valid lower corner
                [8, 8, 8],  # valid upper corner
                [9, 5, 5],  # outside the high z margin
            ]
        )
    }


class VisualizeSwcMasksTest(unittest.TestCase):
    """Tests selection and rendering helpers."""

    @classmethod
    def setUpClass(cls):
        script = (
            Path(__file__).parents[1]
            / "scripts"
            / "visualize_swc_masks.py"
        )
        cls.namespace = runpy.run_path(str(script))

    def test_pick_examples_keeps_only_full_patches(self):
        """Selected SWC centers leave a full patch inside the image."""
        examples = self.namespace["pick_examples"](
            _Dataset(), n=10, brain_id="brain", seed=0
        )
        self.assertEqual(
            set(examples),
            {("brain", (2, 2, 2)), ("brain", (8, 8, 8))},
        )

    def test_overlay_tints_only_masked_pixels(self):
        """The overlay leaves background gray and tints mask pixels red."""
        gray = np.full((2, 2), 0.5, dtype=np.float32)
        mask = np.array([[True, False], [False, False]])
        result = self.namespace["overlay"](gray, mask, alpha=0.5)
        np.testing.assert_allclose(result[1, 1], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(result[0, 0], [0.75, 0.25, 0.25])


if __name__ == "__main__":
    unittest.main()
