"""Tests for the signal-preserving loss module."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock

import torch

from aind_exaspim_image_compression.machine_learning.losses import (
    CompositeDenoisingLoss,
    NoiseStandardizedCountLoss,
    SignalPreservingLoss,
    build_loss,
    charbonnier,
)
from aind_exaspim_image_compression.machine_learning.transforms import (
    LinearClipTransform,
)
from aind_exaspim_image_compression.machine_learning.train import Trainer


class CharbonnierTest(unittest.TestCase):
    """Tests for the Charbonnier penalty."""

    def test_approximates_l1(self):
        """Charbonnier is close to |x| away from zero."""
        d = torch.tensor([3.0, -4.0])
        c = charbonnier(d, eps=1e-3)
        self.assertTrue(
            torch.allclose(c, torch.tensor([3.0, 4.0]), atol=1e-2)
        )


class SignalPreservingLossTest(unittest.TestCase):
    """Tests for SignalPreservingLoss."""

    def test_fg_weight_zero_is_uniform_charbonnier(self):
        """With fg_weight=0 the loss is a plain Charbonnier mean."""
        loss = SignalPreservingLoss(fg_weight=0.0)
        pred = torch.zeros(2, 1, 4, 4, 4)
        target = torch.ones(2, 1, 4, 4, 4)
        fg = torch.ones(2, 1, 4, 4, 4)
        self.assertAlmostEqual(float(loss(pred, target, fg)), 1.0, places=2)

    def test_foreground_error_weighted_more(self):
        """An error on a foreground voxel costs more than on background."""
        loss = SignalPreservingLoss(fg_weight=10.0)
        target = torch.zeros(1, 1, 2, 2, 2)
        fg = torch.zeros(1, 1, 2, 2, 2)
        fg[0, 0, 0, 0, 0] = 1.0

        pred_fg = torch.zeros(1, 1, 2, 2, 2)
        pred_fg[0, 0, 0, 0, 0] = 1.0  # error lands on the foreground voxel
        pred_bg = torch.zeros(1, 1, 2, 2, 2)
        pred_bg[0, 0, 1, 1, 1] = 1.0  # error lands on a background voxel

        self.assertGreater(
            float(loss(pred_fg, target, fg)),
            float(loss(pred_bg, target, fg)),
        )

    def test_gradient_flows(self):
        """Loss is differentiable w.r.t. the prediction."""
        loss = SignalPreservingLoss(fg_weight=5.0)
        pred = torch.zeros(1, 1, 2, 2, 2, requires_grad=True)
        target = torch.ones(1, 1, 2, 2, 2)
        fg = torch.ones(1, 1, 2, 2, 2)
        loss(pred, target, fg).backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.all(pred.grad <= 0))  # move pred toward target


class NoiseStandardizedCountLossTest(unittest.TestCase):
    """Tests count inversion, noise scaling, and saturation masking."""

    @staticmethod
    def _loss(**kwargs):
        """Build a simple count loss with one normalized unit per count."""
        transform = LinearClipTransform(mn=0, mx=100, max_count=100)
        return NoiseStandardizedCountLoss(
            transform,
            sigma_floor=2,
            saturation_margin=0,
            max_count=100,
            **kwargs,
        )

    @staticmethod
    def _metadata(batch_size=1):
        """Return constant-noise metadata for a small batch."""
        return (
            torch.tensor([[0.0, 4.0]]).repeat(batch_size, 1),
            torch.zeros(batch_size),
        )

    def test_standardizes_count_error_by_sigma(self):
        """A ten-count error with sigma two produces error five."""
        loss = self._loss()
        pred = torch.full((1, 1, 2, 2, 2), 0.1)
        target = torch.zeros_like(pred)
        raw = torch.zeros_like(pred)
        noise, offset = self._metadata()
        actual = loss(pred, target, raw, noise, offset)
        self.assertAlmostEqual(float(actual), 5.0, places=4)

    def test_signal_dependent_sigma_reduces_standardized_error(self):
        """Equal count errors cost less where calibrated noise is larger."""
        transform = LinearClipTransform(mn=0, mx=200, max_count=200)
        loss = NoiseStandardizedCountLoss(
            transform, sigma_floor=1, saturation_margin=0, max_count=200
        )
        noise = torch.tensor([[1.0, 0.0]])
        offset = torch.zeros(1)
        raw = torch.zeros(1, 1, 1, 1, 1)
        low = loss(
            torch.tensor([[[[[0.05]]]]]),
            torch.zeros_like(raw),
            raw,
            noise,
            offset,
        )
        high = loss(
            torch.tensor([[[[[0.55]]]]]),
            torch.full_like(raw, 100.0),
            raw,
            noise,
            offset,
        )
        self.assertGreater(float(low), float(high))

    def test_saturation_uses_raw_counts_plus_offset(self):
        """Offset-restored raw sensor values define invalid saturation."""
        loss = self._loss()
        raw = torch.full((1, 1, 1, 1, 1), 90.0)
        valid = loss._valid_mask(raw, torch.tensor([9.0]))
        saturated = loss._valid_mask(raw, torch.tensor([10.0]))
        self.assertTrue(bool(valid.item()))
        self.assertFalse(bool(saturated.item()))

    def test_saturation_dilation_masks_only_local_core_neighborhood(self):
        """One-voxel dilation leaves the more distant halo supervised."""
        loss = self._loss(saturation_dilate=1)
        raw = torch.zeros(1, 1, 5, 5, 5)
        raw[0, 0, 2, 2, 2] = 100
        valid = loss._valid_mask(raw, torch.zeros(1))
        self.assertFalse(bool(valid[0, 0, 1, 2, 2]))
        self.assertFalse(bool(valid[0, 0, 3, 2, 2]))
        self.assertTrue(bool(valid[0, 0, 0, 0, 0]))

    def test_masked_mean_excludes_saturated_error(self):
        """A large saturated-core error does not enter the masked mean."""
        loss = self._loss()
        pred = torch.zeros(1, 1, 1, 1, 2)
        pred[..., 0] = 1.0
        target = torch.zeros_like(pred)
        raw = torch.zeros_like(pred)
        raw[..., 0] = 100
        noise, offset = self._metadata()
        actual = loss(pred, target, raw, noise, offset)
        self.assertAlmostEqual(float(actual), loss.eps, places=6)

    def test_all_saturated_batch_returns_zero_with_finite_gradient(self):
        """The denominator guard handles batches with no valid voxels."""
        loss = self._loss()
        pred = torch.ones(1, 1, 2, 2, 2, requires_grad=True)
        target = torch.zeros_like(pred)
        raw = torch.full_like(pred, 100)
        noise, offset = self._metadata()
        actual = loss(pred, target, raw, noise, offset)
        actual.backward()
        self.assertEqual(float(actual.detach()), 0.0)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_standardized_error_cap_limits_outliers(self):
        """The optional symmetric cap bounds extreme count residuals."""
        loss = self._loss(standardized_error_cap=3)
        pred = torch.ones(1, 1, 1, 1, 1)
        target = torch.zeros_like(pred)
        raw = torch.zeros_like(pred)
        noise, offset = self._metadata()
        self.assertAlmostEqual(
            float(loss(pred, target, raw, noise, offset)), 3.0, places=4
        )

    def test_float32_loss_and_gradient_under_autocast(self):
        """Count inversion and loss remain float32 under AMP."""
        loss = self._loss()
        pred = torch.full(
            (1, 1, 2, 2, 2), 0.1, dtype=torch.bfloat16, requires_grad=True
        )
        target = torch.zeros_like(pred)
        raw = torch.zeros_like(pred)
        noise, offset = self._metadata()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = loss(pred, target, raw, noise, offset)
        actual.backward()
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(pred.grad).all())


class CompositeDenoisingLossTest(unittest.TestCase):
    """Tests configuration modes and weighted loss composition."""

    def setUp(self):
        """Use a linear transform for analytically simple count inversion."""
        self.transform = LinearClipTransform(mn=0, mx=100, max_count=100)
        self.batch = {
            "pred": torch.full((1, 1, 1, 1, 1), 0.1),
            "target": torch.zeros(1, 1, 1, 1, 1),
            "fg": torch.zeros(1, 1, 1, 1, 1),
            "target_counts": torch.zeros(1, 1, 1, 1, 1),
            "raw_counts": torch.zeros(1, 1, 1, 1, 1),
            "noise_params": torch.tensor([[0.0, 4.0]]),
            "offset": torch.zeros(1),
        }

    def _evaluate(self, loss):
        """Evaluate one configured composite on the shared test batch."""
        return loss(
            self.batch["pred"],
            self.batch["target"],
            self.batch["fg"],
            target_counts=self.batch["target_counts"],
            raw_counts=self.batch["raw_counts"],
            noise_params=self.batch["noise_params"],
            offset=self.batch["offset"],
        )

    def test_required_legacy_count_and_combined_modes(self):
        """Each required A/B configuration enables the intended terms."""
        legacy = build_loss(
            {"legacy_weight": 1, "count_weight": 0}, self.transform
        )
        count = build_loss(
            {"legacy_weight": 0, "count_weight": 1}, self.transform
        )
        combined = build_loss(
            {"legacy_weight": 2, "count_weight": 3}, self.transform
        )
        legacy_value = self._evaluate(legacy)
        count_value = self._evaluate(count)
        combined_value = self._evaluate(combined)
        self.assertTrue(
            torch.allclose(
                combined_value, 2 * legacy_value + 3 * count_value
            )
        )
        self.assertTrue(
            torch.allclose(
                combined.last_components["legacy"], 2 * legacy_value
            )
        )
        self.assertTrue(
            torch.allclose(
                combined.last_components["count"], 3 * count_value
            )
        )
        self.assertFalse(legacy.requires_count_metadata)
        self.assertTrue(count.requires_count_metadata)

    def test_count_mode_requires_metadata(self):
        """A count-enabled composite reports missing cache fields clearly."""
        loss = build_loss(
            {"legacy_weight": 0, "count_weight": 1}, self.transform
        )
        with self.assertRaisesRegex(ValueError, "target_counts"):
            loss(self.batch["pred"], self.batch["target"], self.batch["fg"])

    def test_build_loss_resolves_complete_serializable_configuration(self):
        """Defaults are frozen into the configuration stored in checkpoints."""
        loss = build_loss(
            {
                "legacy_weight": 1,
                "count_weight": 0.25,
                "count": {"saturation_dilate": 1},
            },
            self.transform,
        )
        self.assertIsInstance(loss, CompositeDenoisingLoss)
        self.assertEqual(loss.cfg["count"]["sigma_floor"], 2.0)
        self.assertEqual(loss.cfg["count"]["saturation_dilate"], 1)
        self.assertEqual(loss.cfg["count"]["max_count"], 100.0)

    def test_invalid_loss_weights_are_rejected(self):
        """At least one finite nonnegative term must be active."""
        with self.assertRaisesRegex(ValueError, "loss weights"):
            build_loss(
                {"legacy_weight": 0, "count_weight": 0}, self.transform
            )

    def test_trainer_passes_count_metadata_and_checkpoints_loss_config(self):
        """Configured count supervision is wired through trainer persistence."""
        class IdentityModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))
                self.config = {"in_channels": 1}

            def forward(self, x):
                return x * self.scale

        criterion = build_loss(
            {"legacy_weight": 0, "count_weight": 1}, self.transform
        )
        batch = {
            "input": torch.full((1, 1, 1, 1, 1), 0.1),
            "target": torch.zeros(1, 1, 1, 1, 1),
            "foreground": torch.zeros(1, 1, 1, 1, 1),
            "target_counts": torch.zeros(1, 1, 1, 1, 1),
            "raw": torch.zeros(1, 1, 1, 1, 1),
            "noise_params": torch.tensor([[0.0, 4.0]]),
            "offset": torch.zeros(1),
        }
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                directory,
                device="cpu",
                model=IdentityModel(),
                criterion=criterion,
                max_epochs=1,
                use_amp=False,
            )
            try:
                trainer.transform = self.transform
                _, actual = trainer.forward_pass(batch)
                self.assertTrue(torch.isfinite(actual))
                trainer.writer.add_scalar = MagicMock()
                trainer.train_step([batch], epoch=0)
                logged_names = {
                    call.args[0]
                    for call in trainer.writer.add_scalar.call_args_list
                }
                self.assertIn("train_legacy_loss", logged_names)
                self.assertIn("train_count_loss", logged_names)
                trainer.save_model(1, 1.0)
                checkpoint_path = next(
                    trainer.log_dir + "/" + name
                    for name in os.listdir(trainer.log_dir)
                    if name.endswith(".pth")
                )
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                self.assertEqual(checkpoint["loss_config"], criterion.cfg)
            finally:
                trainer.writer.close()


if __name__ == "__main__":
    unittest.main()
