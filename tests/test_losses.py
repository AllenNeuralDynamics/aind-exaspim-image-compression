"""Tests for the signal-preserving loss module."""

import unittest

import torch

from aind_exaspim_image_compression.machine_learning.losses import (
    SignalPreservingLoss,
    charbonnier,
)


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


if __name__ == "__main__":
    unittest.main()
