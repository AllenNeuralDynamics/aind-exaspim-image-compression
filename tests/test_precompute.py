"""Tests for the patch-cache precompute script."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import runpy
import unittest


class _StopAfterConfig(Exception):
    """Stops precompute before allocating cache arrays."""


class PrecomputeConfigTest(unittest.TestCase):
    """Tests persistence of the resolved precompute configuration."""

    def test_writes_complete_config_before_cache_generation(self):
        """Every cache-generation setting is recorded in config.json."""
        script = Path(__file__).parents[1] / "scripts" / "precompute.py"
        namespace = runpy.run_path(str(script))
        precompute = namespace["precompute"]
        settings = {
            "split": "train",
            "cache_dir": "/cache",
            "n_patches": 12,
            "brain_ids_path": "/data/brains.txt",
            "img_prefixes_path": "/data/images.json",
            "segmentation_prefixes_path": "/data/segments.json",
            "offsets_path": "/data/offsets.json",
            "swc_pointers": {"bucket_name": "bucket", "path": "swcs"},
            "transform_cfg": {
                "kind": "asinh",
                "params": {"offset": 0.0, "scale": 32.0},
            },
            "foreground_sampling_rate": 0.5,
            "min_foreground_voxels": 50,
            "min_segmentation_volume": 200,
            "patch_shape": (64, 64, 64),
            "skeleton_radius": 2,
            "segmentation_dilate": 0,
            "sigma_bm4d": 24,
            "reject_incoherent_patches": True,
            "coherence_min_autocorr": 0.4,
            "coherence_max_highfreq_frac": 0.35,
            "coherence_min_segment_voxels": 50,
            "coherence_smooth_sigma": 1.0,
            "coherence_lag": 2,
            "max_resample_attempts": 50,
            "seed": 42,
            "num_workers": None,
        }
        precompute.__globals__.update(settings)
        precompute.__globals__["open_memmap"] = MagicMock(
            side_effect=_StopAfterConfig
        )

        util = precompute.__globals__["util"]
        with patch.object(util, "read_txt", return_value=["brain"]), \
                patch.object(util, "read_json", return_value={"brain": 10}), \
                patch.object(util, "mkdir"), \
                patch.object(util, "write_json") as write_json:
            with self.assertRaises(_StopAfterConfig):
                precompute()

        self.assertEqual(write_json.call_count, 2)
        path, config = write_json.call_args_list[0].args
        self.assertEqual(path, "/cache/config.json")
        expected_keys = set(settings) | {
            "cache_metadata_version",
            "code_version",
            "count_dtype",
            "gat_sigma_multiplier",
            "noise_models",
            "noise_models_path",
            "saturation_margins",
            "seed_stream",
            "teacher_mode",
        }
        self.assertEqual(set(config), expected_keys)
        for key, value in settings.items():
            self.assertEqual(config[key], value)
        self.assertEqual(config["seed_stream"], 0)
        self.assertEqual(config["count_dtype"], "float32")
        self.assertEqual(config["teacher_mode"], "raw_bm4d")
        self.assertEqual(config["gat_sigma_multiplier"], 1.0)
        self.assertEqual(config["sigma_bm4d"], 24)
        self.assertIsNone(config["noise_models"])
        self.assertIsNone(config["noise_models_path"])
        self.assertIsNone(config["saturation_margins"])
        self.assertEqual(config["cache_metadata_version"], 1)
        brain_ids_path, brain_ids = write_json.call_args_list[1].args
        self.assertEqual(brain_ids_path, "/cache/brain_ids.json")
        self.assertEqual(brain_ids, ["brain"])
        self.assertEqual(
            precompute.__globals__["_COUNT_DTYPE"], np.float32
        )
        self.assertEqual(
            precompute.__globals__["open_memmap"].call_args.kwargs["dtype"],
            np.float32,
        )


if __name__ == "__main__":
    unittest.main()
