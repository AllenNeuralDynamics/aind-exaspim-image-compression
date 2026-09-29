"""Tests for detected-cell patch sampling in TrainDataset."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from aind_exaspim_image_compression.machine_learning.data_handling import (
    TrainDataset,
    find_detected_cells_paths,
    resolve_detected_cells_path,
)


def _write_cells_csv(path, xyz):
    with open(path, "w") as f:
        f.write("x,y,z\n")
        for x, y, z in xyz:
            f.write(f"{x},{y},{z}\n")


class CellSamplingTest(unittest.TestCase):
    """Tests cell CSV ingestion, jittered sampling, and patch reads."""

    patch_shape = (8, 8, 8)

    def _dataset(self, xyz, cell_max_shift=None):
        dataset = TrainDataset(
            self.patch_shape, cell_max_shift=cell_max_shift
        )
        dataset.imgs["brain"] = np.arange(
            1 * 1 * 40 * 50 * 60, dtype=np.float32
        ).reshape(1, 1, 40, 50, 60)
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "detected_cells.csv")
            _write_cells_csv(csv_path, xyz)
            dataset._load_cells("brain", csv_path)
        return dataset

    def test_load_cells_flips_xyz_to_zyx(self):
        """CSV (x, y, z) columns are stored as (z, y, x) voxels."""
        dataset = self._dataset([(10, 20, 30), (1.4, 2.6, 3.5)])
        np.testing.assert_array_equal(
            dataset.cells["brain"], np.array([[30, 20, 10], [4, 3, 1]])
        )

    def test_empty_csv_leaves_brain_unregistered(self):
        """A brain with no detected cells is not added to self.cells."""
        dataset = self._dataset([])
        self.assertNotIn("brain", dataset.cells)

    def test_sampled_voxel_stays_within_jitter_radius(self):
        """Jitter never exceeds cell_max_shift on any axis."""
        dataset = self._dataset([(30, 25, 20)], cell_max_shift=(2, 2, 2))
        center = np.array([20, 25, 30])
        for _ in range(50):
            voxel = dataset.sample_cell_voxel("brain")
            self.assertTrue(np.all(np.abs(np.array(voxel) - center) <= 2))

    def test_default_jitter_is_a_quarter_patch(self):
        """cell_max_shift defaults to patch_shape // 4."""
        dataset = self._dataset([(30, 25, 20)])
        np.testing.assert_array_equal(dataset.cell_max_shift, [2, 2, 2])

    def test_boundary_cell_yields_full_patch(self):
        """A cell at the volume corner still reads a full-shape patch."""
        dataset = self._dataset([(0, 0, 0), (59, 49, 39)])
        for _ in range(20):
            voxel = dataset.sample_cell_voxel("brain")
            patch = dataset.read_patch("brain", voxel)
            self.assertEqual(patch.shape, self.patch_shape)

    def test_foreground_sampling_prefers_cells(self):
        """sample_foreground_voxel routes to cells when a brain has them."""
        dataset = self._dataset([(30, 25, 20)], cell_max_shift=(0, 0, 0))
        self.assertEqual(
            dataset.sample_foreground_voxel("brain"), (20, 25, 30)
        )


class InteriorSamplingTest(unittest.TestCase):
    """Tests that interior sampling survives an oversized boundary buffer."""

    patch_shape = (8, 8, 8)

    def _dataset(self, boundary_buffer):
        dataset = TrainDataset(
            self.patch_shape, boundary_buffer=boundary_buffer
        )
        # Short first axis, like SmartSPIM z against an exaSPIM-sized buffer.
        dataset.imgs["brain"] = np.zeros((1, 1, 40, 200, 200), np.float32)
        return dataset

    def test_buffer_larger_than_axis_does_not_raise(self):
        """A buffer exceeding an axis falls back to the in-bounds range."""
        dataset = self._dataset(5000)
        for _ in range(50):
            voxel = dataset.sample_interior_voxel("brain")
            patch = dataset.read_patch("brain", voxel)
            self.assertEqual(patch.shape, self.patch_shape)

    def test_buffer_is_respected_when_it_fits(self):
        """A buffer that fits still keeps samples away from the edges."""
        dataset = self._dataset(60)
        for _ in range(50):
            voxel = dataset.sample_interior_voxel("brain")
            self.assertTrue(60 <= voxel[1] <= 140, voxel)
            self.assertTrue(60 <= voxel[2] <= 140, voxel)


class ResolveDetectedCellsTest(unittest.TestCase):
    """Tests derivation of the detected_cells.csv path from an image path."""

    asset = "s3://bucket/SmartSPIM_675376_2023-06-05_stitched_2026-05-06"
    image = asset + "/image_tile_fusing/OMEZarr/Ex_561_Em_593.zarr"
    cells = asset + "/image_cell_segmentation/Ex_561_Em_593/detected_cells.csv"

    def test_resolves_sibling_cell_segmentation(self):
        """The CSV is the same-channel sibling of the fused image."""
        self.assertEqual(resolve_detected_cells_path(self.image), self.cells)

    def test_tolerates_trailing_multiscale_level(self):
        """A trailing /0 multiscale level does not change the channel."""
        self.assertEqual(
            resolve_detected_cells_path(self.image + "/0"), self.cells
        )

    def test_returns_none_for_other_layouts(self):
        """A path outside the fused SmartSPIM layout resolves to None."""
        self.assertIsNone(
            resolve_detected_cells_path("s3://bucket/exaspim/fused.zarr/0")
        )

    def test_omits_brains_without_a_csv(self):
        """Brains whose CSV does not exist are left out of the mapping."""
        prefixes = {"a": self.image, "b": self.image.replace("675376", "x")}
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, "prefixes.json")
            with open(json_path, "w") as f:
                json.dump(prefixes, f)
            with patch(
                "aind_exaspim_image_compression.machine_learning."
                "data_handling._path_exists",
                side_effect=lambda p: "675376" in p,
            ):
                found = find_detected_cells_paths(["a", "b"], json_path)
        self.assertEqual(found, {"a": self.cells})


if __name__ == "__main__":
    unittest.main()
