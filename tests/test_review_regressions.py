"""Regression tests for inference, persistence, and architecture reviews."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import zarr

from aind_exaspim_image_compression.inference import (
    build_volume_transform,
    load_model,
)
from aind_exaspim_image_compression.machine_learning.data_handling import (
    TrainDataset,
)
from aind_exaspim_image_compression.machine_learning.transforms import (
    build_transform,
)
from aind_exaspim_image_compression.machine_learning.train import Trainer
from aind_exaspim_image_compression.machine_learning.unet3d import (
    DoubleConv,
    UNet,
)
from aind_exaspim_image_compression.utils import img_util


class InferenceRegressionTest(unittest.TestCase):
    """Tests checkpoint reconstruction and inference offset calibration."""

    def test_load_model_reconstructs_constructor_config(self):
        """Loading preserves width, upsampling, and residual mode."""
        model_config = {
            "width_multiplier": 2,
            "trilinear": False,
            "residual": False,
        }
        checkpoint = {
            "model": {"weight": torch.ones(1)},
            "model_config": model_config,
            "transform": {"kind": "asinh", "params": {"scale": 16.0}},
        }
        with patch(
            "aind_exaspim_image_compression.inference.torch.load",
            return_value=checkpoint,
        ), patch(
            "aind_exaspim_image_compression.inference.UNet"
        ) as model_factory:
            loaded, transform = load_model("checkpoint.pth", device="cpu")
        model_factory.assert_called_once_with(**model_config)
        loaded.load_state_dict.assert_called_once_with(checkpoint["model"])
        self.assertEqual(transform.scale, 16.0)

    def test_volume_transform_uses_precomputed_offset(self):
        """A full-tile offset bypasses subvolume-based estimation."""
        base = build_transform(
            {"kind": "asinh", "params": {"offset": 0.0, "scale": 32.0}}
        )
        with patch(
            "aind_exaspim_image_compression.inference.estimate_offset"
        ) as estimate:
            transform = build_volume_transform(base, offset=73.5)
        estimate.assert_not_called()
        self.assertAlmostEqual(transform.offset, 73.5)

    def test_volume_transform_requires_image_without_offset(self):
        """Fallback estimation requires an explicit test image."""
        base = build_transform({"kind": "asinh"})
        with self.assertRaisesRegex(ValueError, "img is required"):
            build_volume_transform(base)

    def test_volume_transform_estimates_directly_from_debug_image(self):
        """Debug fallback uses the supplied subvolume without coarsening."""
        base = build_transform({"kind": "asinh"})
        raw = np.array([0, 10, 20], dtype=np.uint16).reshape(1, 1, 3)
        transform = build_volume_transform(base, raw, percentile=0)
        self.assertAlmostEqual(transform.offset, 10.0)


class ArchitectureRegressionTest(unittest.TestCase):
    """Tests non-default U-Net construction."""

    def test_width_multiplier_is_a_positive_integer(self):
        """Arbitrary fractional width multipliers are rejected explicitly."""
        with self.assertRaisesRegex(ValueError, "positive integer"):
            UNet(width_multiplier=0.3)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            UNet(width_multiplier=0)

    def test_group_norm_groups_divide_channels(self):
        """DoubleConv still chooses valid groups for custom channel counts."""
        block = DoubleConv(1, 9)
        norms = [
            m for m in block.modules()
            if isinstance(m, torch.nn.GroupNorm)
        ]
        self.assertTrue(norms)
        for norm in norms:
            self.assertEqual(norm.num_channels % norm.num_groups, 0)

    def test_checkpoint_saves_config_and_rejects_transform_mismatch(self):
        """Resume cannot silently change the model's intensity mapping."""
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.config = {
                    "width_multiplier": 1,
                    "trilinear": True,
                    "residual": False,
                }

        model = TinyModel()
        checkpoint_transform = build_transform(
            {"kind": "asinh", "params": {"scale": 16.0}}
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                directory,
                device="cpu",
                model=model,
                max_epochs=0,
                use_amp=False,
            )
            try:
                trainer.transform = checkpoint_transform
                trainer.save_model(epoch=0, score=1.0)
                checkpoint_path = os.path.join(
                    trainer.log_dir,
                    next(
                        name
                        for name in os.listdir(trainer.log_dir)
                        if name.endswith(".pth")
                    ),
                )
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                self.assertEqual(checkpoint["model_config"], model.config)

                trainer.load_pretrained_weights(checkpoint_path)
                different_transform = build_transform(
                    {"kind": "asinh", "params": {"scale": 32.0}}
                )
                dataset = SimpleNamespace(transform=different_transform)
                with self.assertRaisesRegex(ValueError, "transform"):
                    trainer.run(dataset, dataset)
            finally:
                trainer.writer.close()


class SkeletonRegressionTest(unittest.TestCase):
    """Tests SWC density validation during ingestion."""

    def test_warns_for_long_edge_but_accepts_3d_diagonal(self):
        """Chebyshev length treats a one-step 3D diagonal as adjacent."""
        dataset = TrainDataset((16, 16, 16), anisotropy=(1, 1, 1))
        swc = {
            "id": np.array([1, 2, 3]),
            "pid": np.array([-1, 1, 2]),
            "xyz": np.array(
                [[1, 2, 3], [2, 3, 4], [4, 3, 4]], dtype=np.float32
            ),
        }
        dataset.swc_reader.read = lambda _: [swc]
        with patch(
            "aind_exaspim_image_compression.machine_learning."
            "data_handling.logger.warning"
        ) as warning:
            dataset._load_swcs("brain", "unused")
        warning.assert_called_once()
        self.assertEqual(warning.call_args.args[1:], ("brain", 1, 2))
        self.assertEqual(dataset.skeletons["brain"].shape, (3, 3))


class ImageUtilityRegressionTest(unittest.TestCase):
    """Tests lazy reads, Zarr 3 writers, and numeric SSIM."""

    def test_n5_reader_returns_tensorstore_without_reading(self):
        """Opening N5 does not materialize the entire volume."""
        sentinel = object()
        opened = type("OpenResult", (), {"result": lambda self: sentinel})()
        with patch.object(img_util.ts, "open", return_value=opened):
            result = img_util._read_n5("/tmp/example.n5")
        self.assertIs(result, sentinel)

    def test_write_zarr_round_trip(self):
        """The basic writer persists a format readable by installed Zarr."""
        image = np.arange(64, dtype=np.uint16).reshape(4, 4, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "image.zarr")
            img_util.write_zarr(image, path, chunks=(1, 1, 2, 2, 2))
            stored = zarr.open(path, mode="r")
            self.assertEqual(stored.metadata.zarr_format, 3)
            np.testing.assert_array_equal(stored[0, 0], image)
            lazy = img_util.read(path)
            np.testing.assert_array_equal(lazy[0, 0], image)

    def test_write_ome_zarr_round_trip(self):
        """OME-Zarr output is persisted in Zarr format 3."""
        image = np.arange(64, dtype=np.uint16).reshape(4, 4, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "image.ome.zarr")
            img_util.write_ome_zarr(
                image,
                path,
                chunks=(1, 1, 2, 2, 2),
                n_levels=2,
                scale=(1, 1, 1.0, 0.748, 0.748),
                translation=(0, 0, 10.0, 20.0, 30.0),
                spatial_unit="micrometer",
            )
            group = zarr.open_group(path, mode="r")
            self.assertEqual(group.metadata.zarr_format, 3)
            self.assertIn("ome", group.attrs)
            multiscales = group.attrs["ome"]["multiscales"]
            dataset_path = multiscales[0]["datasets"][0]["path"]
            np.testing.assert_array_equal(group[dataset_path][0, 0], image)
            transforms = multiscales[0]["datasets"][0][
                "coordinateTransformations"
            ]
            self.assertEqual(
                transforms,
                [
                    {"type": "scale", "scale": [1, 1, 1.0, 0.748, 0.748]},
                    {
                        "type": "translation",
                        "translation": [0, 0, 10.0, 20.0, 30.0],
                    },
                ],
            )
            self.assertEqual(
                multiscales[0]["datasets"][1]["coordinateTransformations"],
                [
                    {"type": "scale", "scale": [1, 1, 2.0, 1.496, 1.496]},
                    {
                        "type": "translation",
                        "translation": [0, 0, 10.5, 20.374, 30.374],
                    },
                ],
            )
            self.assertEqual(
                img_util.get_ome_zarr_level_transform(
                    os.path.join(path, dataset_path)
                ),
                {
                    "scale": (1.0, 1.0, 1.0, 0.748, 0.748),
                    "translation": (0.0, 0.0, 10.0, 20.0, 30.0),
                    "spatial_unit": "micrometer",
                },
            )

    def test_ome_coordinate_with_negative_y_converts_to_voxel(self):
        """Displayed Neuroglancer coordinates map back to level voxels."""
        transform = {
            "scale": (1, 1, 1.0, 0.748, 0.748),
            "translation": (0, 0, 8153.2, -15468.424, 10217.7),
            "spatial_unit": "micrometer",
        }
        voxel_zyx = img_util.ome_zarr_coordinate_to_voxel(
            (22464, -15914, 18711), transform
        )
        self.assertEqual(voxel_zyx, (10558, 4766, 8804))

    def test_ssim_uint16_matches_float_computation(self):
        """Bright uint16 products cannot overflow before local filtering."""
        rng = np.random.default_rng(42)
        image1 = rng.integers(40000, 65000, size=(8, 8, 8), dtype=np.uint16)
        image2 = np.clip(
            image1.astype(np.int32) + rng.integers(-1000, 1000, image1.shape),
            0,
            65535,
        ).astype(np.uint16)
        integer_result = img_util.ssim3D(image1, image2, window_size=3)
        float_result = img_util.ssim3D(
            image1.astype(np.float64),
            image2.astype(np.float64),
            window_size=3,
        )
        self.assertAlmostEqual(integer_result, float_result, places=12)


if __name__ == "__main__":
    unittest.main()
