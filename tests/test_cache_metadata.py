"""Tests for patch-cache provenance and dictionary batching."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from aind_exaspim_image_compression.machine_learning.data_handling import (
    CachedPatchDataset,
    CachedValidateDataset,
    DataLoader,
)


class CacheMetadataTest(unittest.TestCase):
    """Exercises new-cache metadata and legacy compatibility."""

    def _write_cache(self, root, metadata=True, n=2):
        """Create a small count-space cache for one test."""
        cache = Path(root) / "cache"
        cache.mkdir()
        shape = (n, 2, 3, 4)
        np.save(cache / "raw.npy", np.ones(shape, dtype=np.float32))
        np.save(cache / "teacher.npy", np.zeros(shape, dtype=np.float32))
        np.save(cache / "fg.npy", np.zeros(shape, dtype=np.uint8))
        if metadata:
            np.save(cache / "brain_index.npy", np.arange(n, dtype=np.int32) % 2)
            np.save(cache / "center.npy", np.arange(n * 3).reshape(n, 3))
            np.save(cache / "offset.npy", np.arange(n, dtype=np.float32))
            np.save(
                cache / "noise_params.npy",
                np.tile(np.array([1.8, 400], dtype=np.float32), (n, 1)),
            )
            (cache / "brain_ids.json").write_text(json.dumps(["a", "b"]))
        return cache

    def test_new_cache_loads_metadata_into_dictionary_examples(self):
        """Every source field is available alongside model fields."""
        with tempfile.TemporaryDirectory() as root:
            cache = self._write_cache(root, n=1)
            dataset = CachedValidateDataset(str(cache))
            example = dataset[0]

        self.assertTrue(dataset.has_metadata)
        self.assertEqual(dataset.brain_ids, ["a", "b"])
        self.assertEqual(
            set(example),
            {
                "input",
                "target",
                "target_counts",
                "raw",
                "teacher",
                "foreground",
                "brain_index",
                "center",
                "offset",
                "noise_params",
            },
        )
        self.assertEqual(example["center"].shape, (3,))
        self.assertEqual(example["noise_params"].shape, (2,))

    def test_legacy_cache_loads_for_legacy_experiment(self):
        """Missing metadata receives explicit legacy placeholders."""
        with tempfile.TemporaryDirectory() as root:
            cache = self._write_cache(root, metadata=False, n=1)
            dataset = CachedPatchDataset(str(cache))
            example = dataset[0]

        self.assertFalse(dataset.has_metadata)
        self.assertEqual(int(example["brain_index"]), -1)
        self.assertTrue(np.isnan(example["noise_params"]).all())

    def test_count_target_respects_foreground_preservation(self):
        """Count-space supervision matches transformed target construction."""
        with tempfile.TemporaryDirectory() as root:
            cache = self._write_cache(root, n=1)
            np.save(
                cache / "fg.npy",
                np.ones((1, 2, 3, 4), dtype=np.uint8),
            )
            preserved = CachedValidateDataset(
                str(cache), preserve_foreground=True
            )[0]
            denoised = CachedValidateDataset(
                str(cache), preserve_foreground=False
            )[0]

        np.testing.assert_array_equal(
            preserved["target_counts"], preserved["raw"]
        )
        np.testing.assert_array_equal(
            denoised["target_counts"], denoised["teacher"]
        )

    def test_noise_aware_run_rejects_legacy_cache(self):
        """Noise-aware consumers fail clearly rather than using placeholders."""
        with tempfile.TemporaryDirectory() as root:
            cache = self._write_cache(root, metadata=False)
            with self.assertRaisesRegex(ValueError, "noise-aware run"):
                CachedPatchDataset(
                    str(cache), require_noise_metadata=True
                )

    def test_gat_teacher_cache_requires_metadata_automatically(self):
        """A GAT-stamped cache cannot masquerade as a legacy cache."""
        with tempfile.TemporaryDirectory() as root:
            cache = self._write_cache(root, metadata=False)
            (cache / "config.json").write_text(
                json.dumps({"teacher_mode": "gat_bm4d"})
            )
            with self.assertRaisesRegex(ValueError, "noise-aware run"):
                CachedValidateDataset(str(cache))

    def test_partial_metadata_is_rejected(self):
        """Metadata files are an all-or-none cache contract."""
        with tempfile.TemporaryDirectory() as root:
            cache = self._write_cache(root, metadata=False)
            np.save(cache / "offset.npy", np.zeros(2, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "incomplete patch metadata"):
                CachedPatchDataset(str(cache))

    def test_inconsistent_first_dimensions_are_rejected(self):
        """All core and metadata arrays describe the same patch count."""
        with tempfile.TemporaryDirectory() as root:
            cache = self._write_cache(root)
            np.save(cache / "center.npy", np.zeros((1, 3), dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "inconsistent lengths"):
                CachedValidateDataset(str(cache))

    def test_dictionary_collator_preserves_field_shapes_and_dtypes(self):
        """Patch, scalar, center, and noise fields stack by actual shape."""
        with tempfile.TemporaryDirectory() as root:
            cache = self._write_cache(root)
            dataset = CachedValidateDataset(str(cache))
            batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))

        self.assertEqual(batch["input"].shape, (2, 1, 2, 3, 4))
        self.assertEqual(batch["target_counts"].shape, (2, 1, 2, 3, 4))
        self.assertEqual(batch["offset"].shape, (2,))
        self.assertEqual(batch["brain_index"].shape, (2,))
        self.assertEqual(batch["center"].shape, (2, 3))
        self.assertEqual(batch["noise_params"].shape, (2, 2))
        self.assertEqual(batch["brain_index"].dtype, torch.int32)
        self.assertEqual(batch["center"].dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
