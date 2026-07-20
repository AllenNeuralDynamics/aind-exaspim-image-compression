"""Tests for complete, reproducibly shuffled cache epochs."""

import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import torch

from aind_exaspim_image_compression.machine_learning.data_handling import (
    CachedPatchDataset,
    CachedValidateDataset,
    DataLoader,
)
from aind_exaspim_image_compression.machine_learning.train import Trainer


class IdentityTransform:
    """Minimal count-space identity transform for cache-index tests."""

    cfg = {"kind": "identity"}

    def forward(self, values):
        """Returns values as float32 without changing their identity."""
        return np.asarray(values, dtype=np.float32)


class FullCacheDataLoaderTest(unittest.TestCase):
    """Tests cache addressing and deterministic epoch index generation."""

    def setUp(self):
        """Creates a 17-example cache with index-valued one-voxel patches."""
        self.temporary_directory = tempfile.TemporaryDirectory()
        cache_dir = self.temporary_directory.name
        raw = np.arange(17, dtype=np.float32).reshape(17, 1, 1, 1)
        np.save(os.path.join(cache_dir, "raw.npy"), raw)
        np.save(os.path.join(cache_dir, "teacher.npy"), raw + 100)
        np.save(os.path.join(cache_dir, "fg.npy"), raw.astype(np.uint8))
        self.cache_dir = cache_dir
        self.transform = IdentityTransform()

    def tearDown(self):
        """Removes the synthetic cache."""
        self.temporary_directory.cleanup()

    @staticmethod
    def _input_order(loader):
        """Collects the index encoded in each input patch from a loader."""
        return [
            int(value)
            for batch in loader
            for value in batch["input"][:, 0, 0, 0, 0].tolist()
        ]

    def test_cached_dataset_is_index_addressable_and_uses_pool_length(self):
        """Dataset item i reads cache record i and length is the pool size."""
        dataset = CachedPatchDataset(
            self.cache_dir,
            transform=self.transform,
            preserve_foreground=False,
        )

        self.assertEqual(len(dataset), 17)
        example = dataset[12]
        self.assertEqual(float(example["input"].item()), 12.0)
        self.assertEqual(float(example["target"].item()), 112.0)
        self.assertEqual(float(example["foreground"].item()), 1.0)

    def test_shuffled_epoch_is_complete_reproducible_and_epoch_specific(self):
        """Every index appears once with stable per-seed, per-epoch order."""
        dataset = CachedPatchDataset(
            self.cache_dir,
            transform=self.transform,
            preserve_foreground=False,
        )
        loader = DataLoader(
            dataset, batch_size=4, num_workers=0, shuffle=True, seed=42
        )
        loader.set_epoch(3)
        first_order = self._input_order(loader)

        duplicate = DataLoader(
            dataset, batch_size=4, num_workers=0, shuffle=True, seed=42
        )
        duplicate.set_epoch(3)
        self.assertEqual(self._input_order(duplicate), first_order)
        self.assertEqual(sorted(first_order), list(range(17)))
        self.assertEqual(len(set(first_order)), 17)

        loader.set_epoch(4)
        self.assertNotEqual(self._input_order(loader), first_order)

    def test_validation_is_ordered_and_final_partial_batch_is_retained(self):
        """Unshuffled validation keeps order and emits its final example."""
        dataset = CachedValidateDataset(
            self.cache_dir,
            transform=self.transform,
            preserve_foreground=False,
        )
        loader = DataLoader(
            dataset, batch_size=4, num_workers=0, shuffle=False
        )
        batches = list(loader)
        order = [
            int(value)
            for batch in batches
            for value in batch["input"][:, 0, 0, 0, 0].tolist()
        ]

        self.assertEqual(order, list(range(17)))
        self.assertEqual(len(batches), 5)
        self.assertEqual(batches[-1]["input"].shape[0], 1)


class TrainerShuffleTest(unittest.TestCase):
    """Tests Trainer wiring for shuffled training and ordered validation."""

    def test_gradient_clipping_requires_positive_finite_norm(self):
        """Invalid clipping thresholds fail before training starts."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                Trainer(
                    directory,
                    device="cpu",
                    model=torch.nn.Linear(1, 1),
                    max_epochs=1,
                    use_amp=False,
                    max_grad_norm=0,
                )

    def test_gradient_clipping_unscales_before_optimizer_step(self):
        """Configured clipping sees unscaled gradients before the update."""
        model = torch.nn.Linear(1, 1)
        values = torch.ones((1, 1, 1, 1, 1))
        batch = {
            "input": values,
            "target": torch.zeros_like(values),
            "foreground": torch.zeros_like(values),
        }
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                directory,
                device="cpu",
                model=model,
                max_epochs=1,
                use_amp=False,
                max_grad_norm=1.0,
            )
            try:
                with (
                    patch.object(
                        trainer.scaler,
                        "unscale_",
                        wraps=trainer.scaler.unscale_,
                    ) as unscale,
                    patch(
                        "aind_exaspim_image_compression.machine_learning."
                        "train.torch.nn.utils.clip_grad_norm_",
                        wraps=torch.nn.utils.clip_grad_norm_,
                    ) as clip,
                ):
                    trainer.train_step([batch], epoch=0)

                unscale.assert_called_once_with(trainer.optimizer)
                clip.assert_called_once()
                self.assertEqual(clip.call_args.kwargs["max_norm"], 1.0)
            finally:
                trainer.writer.close()

    def test_training_and_validation_amp_are_configured_separately(self):
        """Training can use AMP while validation remains full precision."""
        model = torch.nn.Linear(1, 1)
        values = torch.zeros((1, 1, 1, 1, 1))
        batch = {
            "input": values,
            "target": values,
            "foreground": values,
        }
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                directory,
                device="cpu",
                model=model,
                max_epochs=1,
                use_amp=True,
                use_amp_validation=False,
            )
            try:
                with patch(
                    "aind_exaspim_image_compression.machine_learning."
                    "train.torch.autocast",
                    side_effect=[nullcontext(), nullcontext()],
                ) as autocast:
                    trainer.forward_pass(
                        batch, use_amp=trainer.use_amp
                    )
                    trainer.forward_pass(
                        batch,
                        use_amp=trainer.use_amp_validation,
                    )

                self.assertTrue(autocast.call_args_list[0].kwargs["enabled"])
                self.assertFalse(
                    autocast.call_args_list[1].kwargs["enabled"]
                )
            finally:
                trainer.writer.close()

    def test_trainer_sets_epoch_and_persists_seed(self):
        """Trainer passes its seed, sets every epoch, and saves the seed."""
        model = torch.nn.Linear(1, 1)
        dataset = SimpleNamespace(
            patch_shape=(1, 1, 1), transform=IdentityTransform()
        )
        train_loader = MagicMock()
        val_loader = MagicMock()

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                directory,
                device="cpu",
                model=model,
                max_epochs=2,
                use_amp=False,
                num_workers=0,
                seed=42,
            )
            trainer.train_step = MagicMock(return_value=1.0)
            trainer.validate_step = MagicMock(return_value=(1.0, 2.0, False))
            trainer.scheduler.step = MagicMock()
            try:
                with patch(
                    "aind_exaspim_image_compression.machine_learning."
                    "train.DataLoader",
                    side_effect=[train_loader, val_loader],
                ) as loader_factory:
                    trainer.run(dataset, dataset)

                self.assertEqual(
                    loader_factory.call_args_list,
                    [
                        call(
                            dataset,
                            batch_size=16,
                            num_workers=0,
                            prefetch=2,
                            shuffle=True,
                            seed=42,
                        ),
                        call(
                            dataset,
                            batch_size=16,
                            num_workers=0,
                            prefetch=2,
                            shuffle=False,
                        ),
                    ],
                )
                self.assertEqual(
                    train_loader.set_epoch.call_args_list, [call(0), call(1)]
                )

                trainer.save_config({})
                with open(
                    os.path.join(trainer.log_dir, "config.json"),
                    encoding="utf-8",
                ) as file:
                    config = json.load(file)
                    self.assertEqual(config["seed"], 42)
                    self.assertFalse(config["use_amp"])
                    self.assertFalse(config["use_amp_validation"])
                    self.assertIsNone(config["max_grad_norm"])
            finally:
                trainer.writer.close()


if __name__ == "__main__":
    unittest.main()
