"""
Loss functions for signal-preserving denoising.

"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class NoiseStandardizedCountLoss(nn.Module):
    """Charbonnier loss in count space standardized by calibrated noise.

    Saturated and nearly saturated voxels are excluded using the original raw
    sensor counts. The invalid mask may be dilated locally, but unsaturated
    halo voxels outside that radius remain supervised.
    """

    def __init__(
        self,
        transform,
        sigma_floor=2.0,
        saturation_margin=64,
        saturation_dilate=0,
        max_count=65535.0,
        standardized_error_cap=None,
        eps=1e-3,
    ):
        """Configure count inversion, noise scaling, and saturation masking."""
        super().__init__()
        self.transform = transform
        self.sigma_floor = float(sigma_floor)
        self.saturation_margin = float(saturation_margin)
        self.saturation_dilate = int(saturation_dilate)
        self.max_count = float(max_count)
        self.standardized_error_cap = (
            None
            if standardized_error_cap is None
            else float(standardized_error_cap)
        )
        self.eps = float(eps)
        if not math.isfinite(self.sigma_floor) or self.sigma_floor <= 0:
            raise ValueError("sigma_floor must be finite and positive")
        if (
            not math.isfinite(self.saturation_margin)
            or self.saturation_margin < 0
            or self.saturation_margin >= self.max_count
        ):
            raise ValueError(
                "saturation_margin must be finite, nonnegative, and below "
                "max_count"
            )
        if self.saturation_dilate < 0:
            raise ValueError("saturation_dilate must be nonnegative")
        if not math.isfinite(self.max_count) or self.max_count <= 0:
            raise ValueError("max_count must be finite and positive")
        if self.standardized_error_cap is not None and (
            not math.isfinite(self.standardized_error_cap)
            or self.standardized_error_cap <= 0
        ):
            raise ValueError(
                "standardized_error_cap must be finite and positive"
            )
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")

    def _valid_mask(self, raw_counts, offset):
        """Return unsaturated voxels, optionally dilating invalid cores."""
        batch_size = raw_counts.shape[0]
        offset = offset.reshape(batch_size, *([1] * (raw_counts.ndim - 1)))
        threshold = self.max_count - self.saturation_margin
        invalid = raw_counts + offset >= threshold
        if self.saturation_dilate:
            if raw_counts.ndim != 5:
                raise ValueError(
                    "saturation dilation requires 3D batches shaped "
                    "(B, C, D, H, W)"
                )
            radius = self.saturation_dilate
            kernel_size = 2 * radius + 1
            invalid = F.max_pool3d(
                invalid.float(),
                kernel_size=kernel_size,
                stride=1,
                padding=radius,
            ) > 0
        return ~invalid

    def forward(
        self,
        pred_transformed,
        target_counts,
        raw_counts,
        noise_params,
        offset,
    ):
        """Compute the properly masked noise-standardized count loss."""
        device_type = pred_transformed.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            pred_transformed = pred_transformed.float()
            target_counts = target_counts.float()
            raw_counts = raw_counts.float()
            noise_params = noise_params.float()
            offset = offset.float()
            batch_size = pred_transformed.shape[0]
            if noise_params.shape != (batch_size, 2):
                raise ValueError("noise_params must have shape (B, 2)")
            if offset.numel() != batch_size:
                raise ValueError("offset must contain one value per sample")
            if target_counts.shape != pred_transformed.shape:
                raise ValueError(
                    "target_counts must have the same shape as prediction"
                )
            if raw_counts.shape != pred_transformed.shape:
                raise ValueError(
                    "raw_counts must have the same shape as prediction"
                )

            pred_counts = self.transform.inverse_tensor(pred_transformed)
            broadcast_shape = (batch_size,) + (1,) * (
                target_counts.ndim - 1
            )
            slope = noise_params[:, 0].reshape(broadcast_shape)
            intercept = noise_params[:, 1].reshape(broadcast_shape)
            variance = slope * torch.clamp(target_counts, min=0.0) + intercept
            sigma = torch.sqrt(
                torch.clamp(variance, min=self.sigma_floor ** 2)
            )
            standardized_error = (pred_counts - target_counts) / sigma
            if self.standardized_error_cap is not None:
                standardized_error = torch.clamp(
                    standardized_error,
                    min=-self.standardized_error_cap,
                    max=self.standardized_error_cap,
                )
            penalty = charbonnier(standardized_error, self.eps)
            valid = self._valid_mask(raw_counts, offset)
            valid_float = valid.float()
            denominator = torch.clamp(valid_float.sum(), min=1.0)
            return (valid_float * penalty).sum() / denominator


class CompositeDenoisingLoss(nn.Module):
    """Weighted sum of legacy transform-domain and count-space losses."""

    def __init__(
        self,
        legacy_loss,
        count_loss,
        legacy_weight=1.0,
        count_weight=0.0,
        cfg=None,
    ):
        """Configure legacy-only, count-only, or combined supervision."""
        super().__init__()
        self.legacy_loss = legacy_loss
        self.count_loss = count_loss
        self.legacy_weight = float(legacy_weight)
        self.count_weight = float(count_weight)
        self.cfg = copy.deepcopy(cfg) if cfg is not None else None
        self.last_components = {}

    @property
    def fg_weight(self):
        """Expose the legacy foreground weight for checkpoint compatibility."""
        return self.legacy_loss.fg_weight

    @property
    def requires_count_metadata(self):
        """Whether this configuration consumes count-space cache metadata."""
        return self.count_weight > 0

    def forward(
        self,
        pred,
        target,
        fg_mask,
        target_counts=None,
        raw_counts=None,
        noise_params=None,
        offset=None,
    ):
        """Evaluate and combine enabled loss terms."""
        total = pred.new_zeros((), dtype=torch.float32)
        legacy_component = pred.new_zeros((), dtype=torch.float32)
        count_component = pred.new_zeros((), dtype=torch.float32)
        if self.legacy_weight:
            legacy_component = self.legacy_weight * self.legacy_loss(
                pred, target, fg_mask
            )
            total = total + legacy_component
        if self.count_weight:
            required = {
                "target_counts": target_counts,
                "raw_counts": raw_counts,
                "noise_params": noise_params,
                "offset": offset,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "count loss requires batch fields: " + ", ".join(missing)
                )
            count_component = self.count_weight * self.count_loss(
                pred,
                target_counts,
                raw_counts,
                noise_params,
                offset,
            )
            total = total + count_component
        self.last_components = {
            "legacy": legacy_component.detach(),
            "count": count_component.detach(),
        }
        return total


def build_loss(cfg, transform):
    """Build a fully resolved composite denoising loss from configuration."""
    cfg = copy.deepcopy(cfg or {})
    unknown = set(cfg) - {"legacy_weight", "count_weight", "legacy", "count"}
    if unknown:
        raise ValueError(
            "loss configuration has unknown fields: "
            + ", ".join(sorted(unknown))
        )
    legacy_weight = float(cfg.get("legacy_weight", 1.0))
    count_weight = float(cfg.get("count_weight", 0.0))
    if any(
        not math.isfinite(weight) or weight < 0
        for weight in (legacy_weight, count_weight)
    ) or legacy_weight + count_weight <= 0:
        raise ValueError(
            "loss weights must be finite and nonnegative with a positive sum"
        )

    legacy_overrides = cfg.get("legacy", {})
    count_overrides = cfg.get("count", {})
    if not isinstance(legacy_overrides, dict):
        raise ValueError("legacy loss configuration must be an object")
    if not isinstance(count_overrides, dict):
        raise ValueError("count loss configuration must be an object")
    unknown_legacy = set(legacy_overrides) - {"fg_weight", "eps"}
    unknown_count = set(count_overrides) - {
        "sigma_floor",
        "saturation_margin",
        "saturation_dilate",
        "max_count",
        "standardized_error_cap",
        "eps",
    }
    if unknown_legacy or unknown_count:
        unknown_fields = sorted(unknown_legacy | unknown_count)
        raise ValueError(
            "loss term configuration has unknown fields: "
            + ", ".join(unknown_fields)
        )

    legacy_cfg = {"fg_weight": 20.0, "eps": 1e-3}
    legacy_cfg.update(legacy_overrides)
    count_cfg = {
        "sigma_floor": 2.0,
        "saturation_margin": 64,
        "saturation_dilate": 0,
        "max_count": float(transform.max_count),
        "standardized_error_cap": None,
        "eps": 1e-3,
    }
    count_cfg.update(count_overrides)
    resolved = {
        "legacy_weight": legacy_weight,
        "count_weight": count_weight,
        "legacy": legacy_cfg,
        "count": count_cfg,
    }
    legacy_loss = SignalPreservingLoss(**legacy_cfg)
    count_loss = NoiseStandardizedCountLoss(transform, **count_cfg)
    return CompositeDenoisingLoss(
        legacy_loss,
        count_loss,
        legacy_weight=legacy_weight,
        count_weight=count_weight,
        cfg=resolved,
    )
