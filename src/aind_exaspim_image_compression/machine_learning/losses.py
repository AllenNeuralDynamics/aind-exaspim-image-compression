"""
Loss functions for signal-preserving denoising.

"""

import torch
import torch.nn as nn


def charbonnier(diff, eps=1e-3):
    """
    Evaluates the Charbonnier penalty, a smooth approximation of the L1 norm.

    Parameters
    ----------
    diff : torch.Tensor
        Difference tensor.
    eps : float, optional
        Smoothing constant. Default is 1e-3.

    Returns
    -------
    torch.Tensor
        Elementwise Charbonnier penalty.
    """
    return torch.sqrt(diff * diff + eps * eps)


class SignalPreservingLoss(nn.Module):
    """
    Foreground-weighted Charbonnier loss.

    Upweights foreground voxels so that sparse, bright neurites are not
    drowned out by the background during training. Operates in the transform
    domain: because a compressive transform shrinks the bright tail, a fixed
    error here is a larger error in counts, i.e. this enforces relative
    (Weber) precision. If absolute count fidelity is required, weight by the
    transform Jacobian or add a count-space term (deferred; see the
    implementation plan and the operating-point decision).

    Attributes
    ----------
    fg_weight : float
        Extra weight applied to foreground voxels (0 disables weighting, so
        the loss reduces to a plain Charbonnier mean).
    eps : float
        Charbonnier smoothing constant.
    """

    def __init__(self, fg_weight=20.0, eps=1e-3):
        """
        Instantiates a SignalPreservingLoss.

        Parameters
        ----------
        fg_weight : float, optional
            Extra weight applied to foreground voxels. Default is 20.0.
        eps : float, optional
            Charbonnier smoothing constant. Default is 1e-3.
        """
        super().__init__()
        self.fg_weight = float(fg_weight)
        self.eps = float(eps)

    def forward(self, pred, target, fg_mask):
        """
        Computes the foreground-weighted Charbonnier loss.

        Parameters
        ----------
        pred : torch.Tensor
            Model prediction in the transform domain.
        target : torch.Tensor
            Target in the transform domain.
        fg_mask : torch.Tensor
            Foreground mask (0/1) with the same shape as "pred".

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        weight = 1.0 + self.fg_weight * fg_mask
        return (weight * charbonnier(pred - target, self.eps)).mean()
