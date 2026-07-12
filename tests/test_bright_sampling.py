"""Tests for weighted, ROI-aware bright-brain patch sampling."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from aind_exaspim_image_compression.machine_learning.data_handling import (
    TrainDataset,
    resolve_brain_sampling_weights,
    validate_sampling_regions,
)


class BrightSamplingTest(unittest.TestCase):
    """Checks Part 4 sampling controls without cloud-backed images."""

    @staticmethod
    def _dataset(**kwargs):
        """Build a sampler around a synthetic 100-cube image shape."""
        dataset = TrainDataset(
            (10, 10, 10),
            boundary_buffer=0,
            prefetch_foreground_sampling=3,
            **kwargs,
        )
        dataset.imgs["bright"] = SimpleNamespace(shape=(1, 1, 100, 100, 100))
        return dataset

    def test_brain_weights_are_normalized_and_used(self):
        """Configured weight 4 produces an 80/20 requested distribution."""
        weights, distribution = resolve_brain_sampling_weights(
            ["ordinary", "bright"], {"bright": 4}
        )
        self.assertEqual(weights, {"ordinary": 1.0, "bright": 4.0})
        self.assertAlmostEqual(distribution["bright"], 0.8)

        dataset = TrainDataset(
            (10, 10, 10), brain_sampling_weights={"bright": 4}
        )
        dataset.imgs["ordinary"] = object()
        dataset.imgs["bright"] = object()
        with patch(
            "aind_exaspim_image_compression.machine_learning."
            "data_handling.random.choices",
            return_value=["bright"],
        ) as choices:
            self.assertEqual(dataset.sample_brain(), "bright")
        self.assertEqual(choices.call_args.kwargs["weights"], [1.0, 4.0])

    def test_invalid_brain_weight_is_rejected(self):
        """Nonpositive sampling weights cannot silently drop a brain."""
        with self.assertRaisesRegex(ValueError, "positive"):
            resolve_brain_sampling_weights(["a"], {"a": 0})

    def test_roi_sampling_uses_half_open_level_zero_bounds(self):
        """Interior centers stay inside the selected spatial region."""
        dataset = self._dataset(
            sampling_rois={
                "bright": [
                    {"start": [20, 30, 40], "stop": [30, 40, 50], "weight": 2}
                ]
            }
        )
        for _ in range(50):
            center = np.asarray(dataset.sample_interior_voxel("bright"))
            self.assertTrue(np.all(center >= [20, 30, 40]))
            self.assertTrue(np.all(center < [30, 40, 50]))

    def test_training_patches_do_not_intersect_heldout_blocks(self):
        """Patch bounds, rather than only centers, are excluded."""
        dataset = self._dataset(
            heldout_regions={
                "bright": [{"start": [40, 40, 40], "stop": [60, 60, 60]}]
            }
        )
        for _ in range(100):
            center = np.asarray(dataset.sample_interior_voxel("bright"))
            start = center - 5
            stop = start + 10
            intersects = np.all(start < 60) and np.all(stop > 40)
            self.assertFalse(intersects)

    def test_validation_can_include_heldout_blocks(self):
        """Held-out exclusion is explicitly disabled for validation sampling."""
        dataset = self._dataset(
            sampling_rois={
                "bright": [{"start": [45, 45, 45], "stop": [55, 55, 55]}]
            },
            heldout_regions={
                "bright": [{"start": [40, 40, 40], "stop": [60, 60, 60]}]
            },
            exclude_heldout=False,
        )
        center = dataset.sample_interior_voxel("bright")
        self.assertTrue(np.all(np.asarray(center) >= 45))

    def test_bright_mixture_dispatches_requested_condition(self):
        """Bright brains bypass generic sampling for the configured mixture."""
        dataset = self._dataset(
            bright_sampling_weights={"bright": {"halo": 1}}
        )
        with patch.object(
            dataset, "sample_bright_condition_voxel", return_value=(20, 21, 22)
        ) as sample_condition:
            center = dataset.sample_voxel("bright")
        self.assertEqual(center, (20, 21, 22))
        sample_condition.assert_called_once_with("bright", "halo")

    def test_condition_scoring_separates_background_and_bright_unsaturated(self):
        """Candidate scoring selects low background and high unsaturated signal."""
        dataset = self._dataset()
        candidates = [
            ((20, 20, 20), np.full((10, 10, 10), 10, dtype=np.float32)),
            ((30, 30, 30), np.full((10, 10, 10), 1000, dtype=np.float32)),
            ((40, 40, 40), np.full((10, 10, 10), 65535, dtype=np.float32)),
        ]
        with patch.object(dataset, "_bright_candidates", return_value=candidates):
            self.assertEqual(
                dataset.sample_bright_condition_voxel("bright", "background"),
                (20, 20, 20),
            )
            self.assertEqual(
                dataset.sample_bright_condition_voxel(
                    "bright", "bright_unsaturated"
                ),
                (30, 30, 30),
            )

    def test_region_schema_rejects_invalid_bounds(self):
        """Malformed spatial configurations fail before cache generation."""
        with self.assertRaisesRegex(ValueError, "stop must exceed start"):
            validate_sampling_regions(
                {"bright": [{"start": [1, 2, 3], "stop": [1, 4, 5]}]}
            )


if __name__ == "__main__":
    unittest.main()
