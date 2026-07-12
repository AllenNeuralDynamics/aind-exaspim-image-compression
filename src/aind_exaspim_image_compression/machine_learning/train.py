"""
Created on Fri Jan 3 12:30:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code used to train neural network to denoise images.

"""

from contextlib import nullcontext
from datetime import datetime
from numcodecs import blosc
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from skimage import io

from aind_exaspim_image_compression.machine_learning.unet3d import UNet
from aind_exaspim_image_compression.machine_learning.data_handling import DataLoader
from aind_exaspim_image_compression.machine_learning.losses import (
    CompositeDenoisingLoss,
    SignalPreservingLoss,
)
from aind_exaspim_image_compression.machine_learning.metrics import (
    aggregate_stratified_metrics,
    evaluate_stratified_example,
    select_checkpoint,
)
from aind_exaspim_image_compression.utils import img_util, util


def save_mip_png(path, img, low_pct=1.0, high_pct=99.9):
    """
    Writes a 3D volume as a contrast-stretched 8-bit PNG for easy viewing.

    The volume is reduced to 2D with a maximum-intensity projection along the
    z-axis (axis 0), then percentile-normalized to uint8 so that dim neurites
    are visible in a standard image viewer.

    Parameters
    ----------
    path : str
        Output path for the PNG file.
    img : numpy.ndarray
        3D image volume with shape (D, H, W) in raw counts.
    low_pct : float, optional
        Lower percentile mapped to black. The default is 1.0.
    high_pct : float, optional
        Upper percentile mapped to white. The default is 99.9.
    """
    mip = img.max(axis=0).astype(np.float32)
    lo, hi = np.percentile(mip, (low_pct, high_pct))
    if hi <= lo:
        hi = lo + 1.0
    mip = np.clip((mip - lo) / (hi - lo), 0.0, 1.0)
    io.imsave(path, np.rint(mip * 255).astype(np.uint8), check_contrast=False)


class Trainer:

    def __init__(
        self,
        output_dir,
        batch_size=16,
        device="cuda",
        lr=1e-3,
        max_epochs=400,
        model=None,
        use_amp=True,
        checkpoint_weights=None,
        checkpoint_selection=None,
        fg_weight=20.0,
        criterion=None,
        num_workers=None,
        prefetch=2,
        val_every=1,
    ):
        """
        Instantiates a Trainer object.

        Parameters
        ----------
        output_dir : str
            Directory that model checkpoints and tensorboard are written to.
        batch_size : int, optional
            Number of samples per batch during training. Default is 16.
        device : str, optional
            GPU device that model is trained on. Default is "cuda".
        lr : float, optional
            Learning rate. Default is 1e-3.
        max_epochs : int, optional
            Maximum number of training epochs. Default is 400.
        model : None or nn.Module, optional
            Model to be trained on the given datasets. Default is None.
        use_amp : bool, optional
            Indication of whether to use mixed precision. Default is True.
        criterion : torch.nn.Module, optional
            Configured denoising criterion. When omitted, the legacy
            SignalPreservingLoss is constructed from ``fg_weight``.
        checkpoint_selection : dict, optional
            Legacy or constraint-first checkpoint-selection configuration.
            Default is legacy additive scoring.
        val_every : int, optional
            Run validation (and checkpoint selection) every this many epochs;
            the final epoch is always validated. The count-space metrics are
            CPU-bound, so a large validation set is only cheap if it is not run
            every epoch. Default is 1 (validate every epoch).
        """
        # Initializations
        exp_name = "session-" + datetime.today().strftime("%Y%m%d_%H%M")
        log_dir = os.path.join(output_dir, exp_name)
        util.mkdir(log_dir)

        # Instance attributes
        self.batch_size = batch_size
        self.device = device
        self.max_epochs = max_epochs
        self.log_dir = log_dir
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.val_every = max(1, int(val_every))
        self.run_config = None

        self.codec = blosc.Blosc(cname="zstd", clevel=5, shuffle=blosc.SHUFFLE)
        self.criterion = (
            criterion
            if criterion is not None
            else SignalPreservingLoss(fg_weight=fg_weight)
        )
        self.loss_config = getattr(self.criterion, "cfg", None)
        if self.loss_config is None:
            self.loss_config = {
                "legacy_weight": 1.0,
                "count_weight": 0.0,
                "legacy": {
                    "fg_weight": self.criterion.fg_weight,
                    "eps": self.criterion.eps,
                },
                "count": {
                    "sigma_floor": 2.0,
                    "saturation_margin": 64,
                    "saturation_dilate": 0,
                    "max_count": 65535.0,
                    "standardized_error_cap": None,
                    "eps": 1e-3,
                },
            }
        self.checkpoint_weights = checkpoint_weights
        self.checkpoint_selection = (
            {"mode": "legacy"}
            if checkpoint_selection is None
            else checkpoint_selection
        )
        self.last_checkpoint_selection = None
        self.best_score = np.inf
        self.model = model.to(device) if model else UNet().to(device)
        self._resume_transform_cfg = None
        self._resume_run_config = None
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        # T_max spans the whole run so the cosine anneals once. With a small
        # T_max the LR returns to its peak every 2*T_max epochs, and each
        # return destabilized training (growing periodic loss spikes).
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        self.writer = SummaryWriter(log_dir=log_dir)

        if use_amp:
            self.autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            self.autocast = nullcontext()

        # Scale the loss before backward so small float16 gradients do not
        # underflow (and are unscaled before the step). Disabled => no-op, so
        # the same code path is correct with and without AMP.
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Core Routines ---
    def run(self, train_dataset, val_dataset):
        """
        Runs the full training and validation loop.

        Parameters
        ----------
        train_dataset : TrainDataset
            Dataset used for training.
        val_dataset : ValidateDataset
            Dataset used for validation.
        """
        # Initializations
        print("Experiment:", os.path.basename(os.path.normpath(self.log_dir)))
        if self._resume_transform_cfg is not None:
            train_cfg = getattr(train_dataset.transform, "cfg", None)
            val_cfg = getattr(val_dataset.transform, "cfg", None)
            if train_cfg != self._resume_transform_cfg:
                raise ValueError(
                    "resume checkpoint transform does not match the training "
                    "dataset transform"
                )
            if val_cfg != self._resume_transform_cfg:
                raise ValueError(
                    "resume checkpoint transform does not match the validation "
                    "dataset transform"
                )
        self.transform = train_dataset.transform
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch=self.prefetch,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch=self.prefetch,
        )

        # Main
        self.best_score = np.inf
        for epoch in range(self.max_epochs):
            # Train
            train_loss = self.train_step(train_dataloader, epoch)

            # Validate every val_every epochs (and always on the final epoch);
            # the count-space metrics are CPU-bound, so a large validation set
            # is only cheap when it is not run every epoch.
            is_last = epoch == self.max_epochs - 1
            if epoch % self.val_every == 0 or is_last:
                val_loss, val_cratio, is_best = self.validate_step(
                    val_dataloader, epoch
                )
                suffix = " - New Best!" if is_best else ""
                s = f"Epoch {epoch}:  train_loss={train_loss},  val_loss={val_loss}, val_cratio={val_cratio}" + suffix
            else:
                s = f"Epoch {epoch}:  train_loss={train_loss}"
            print(s)

            # Step scheduler
            self.scheduler.step()

    def train_step(self, train_dataloader, epoch):
        """
        Performs a single training epoch over the provided DataLoader.

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        epoch : int
            Current training epoch.

        Returns
        -------
        loss : float
            Average loss over the training epoch.
        """
        losses = list()
        self.model.train()
        for batch in train_dataloader:
            # Forward pass
            hat_y, loss = self.forward_pass(batch)

            # Backward pass (loss-scaled for AMP stability)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Store loss for tensorboard
            losses.append(float(loss.detach().cpu()))
        self.writer.add_scalar("train_loss", np.mean(losses), epoch)
        return np.mean(losses)

    def validate_step(self, val_dataloader, epoch):
        """
        Validates the model over the provided DataLoader.

        Parameters
        ----------
        val_dataloader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        epoch : int
            Current training epoch.

        Returns
        -------
        loss : float
            Average loss over the validation dataset.
        cratio : float
            Average compression ratio over the validation dataset.
        is_best : bool
            Indication of whether the model is the best so far.
        """
        # Skip if there are no validation examples
        if len(val_dataloader.dataset) == 0:
            return float("nan"), float("nan"), False

        losses = list()
        cratios = list()
        metric_records = list()
        with torch.no_grad():
            self.model.eval()
            for batch in val_dataloader:
                # Run model
                hat_y, loss = self.forward_pass(batch)

                # Evaluate result
                losses.append(loss.detach().cpu())
                cratios.extend(self.compute_cratios(hat_y))
                metric_records.extend(
                    self.compute_metrics(
                        hat_y,
                        batch,
                        getattr(val_dataloader.dataset, "brain_ids", []),
                    )
                )

        # Aggregate results
        loss = float(np.mean(losses))
        cratio = float(np.median(cratios))
        metric_report = aggregate_stratified_metrics(metric_records)
        agg = metric_report["overall"]
        selection = select_checkpoint(
            metric_report,
            cratio,
            config=self.checkpoint_selection,
            legacy_weights=self.checkpoint_weights,
        )
        score = selection["score"]
        self.last_checkpoint_selection = selection
        metric_report["checkpoint_selection"] = selection

        # Log results
        self.writer.add_scalar("val_loss", loss, epoch)
        self.writer.add_scalar("val_cratio", cratio, epoch)
        self.writer.add_scalar("val_score", score, epoch)
        self.writer.add_scalar(
            "val_checkpoint_constraints_valid",
            float(selection["valid"]),
            epoch,
        )
        if np.isfinite(selection["objective_value"]):
            self.writer.add_scalar(
                "val_checkpoint_objective",
                selection["objective_value"],
                epoch,
            )
        for name, value in agg.items():
            if np.isscalar(value) and np.isfinite(value):
                self.writer.add_scalar(f"val_{name}", value, epoch)
        self._log_stratified_metrics(metric_report, epoch)
        util.write_json(
            os.path.join(
                self.log_dir, f"validation-metrics-epoch-{epoch}.json"
            ),
            metric_report,
        )

        # Save every validated checkpoint so the best can be chosen offline.
        # Skip epoch 0: the untrained net emits a near-constant, trivially
        # compressible volume whose cratio-weighted score would beat every
        # trained checkpoint despite its high loss. is_best is tracked only for
        # the "New Best!" log line.
        is_best = epoch > 0 and selection["valid"] and score < self.best_score
        if is_best:
            self.best_score = score
        if epoch > 0:
            self.save_model(epoch, score)
        return loss, cratio, is_best

    def forward_pass(self, batch):
        """
        Performs a forward pass through the model and computes loss.

        Parameters
        ----------
        batch : dict[str, torch.Tensor]
            Dictionary containing model fields and optional count metadata.

        Returns
        -------
        hat_y : torch.Tensor
            Model predictions.
        loss : torch.Tensor
            Computed loss value.
        """
        with self.autocast:
            x = batch["input"].to(self.device)
            y = batch["target"].to(self.device)
            fg_mask = batch["foreground"].to(self.device)
            hat_y = self.model(x)
            if isinstance(self.criterion, CompositeDenoisingLoss):
                optional = {
                    name: batch[name].to(self.device)
                    for name in (
                        "target_counts", "raw", "noise_params", "offset"
                    )
                    if name in batch
                }
                loss = self.criterion(
                    hat_y,
                    y,
                    fg_mask,
                    target_counts=optional.get("target_counts"),
                    raw_counts=optional.get("raw"),
                    noise_params=optional.get("noise_params"),
                    offset=optional.get("offset"),
                )
            else:
                loss = self.criterion(hat_y, y, fg_mask)
            return hat_y, loss

    # --- Helpers ---
    def compute_cratios(self, imgs):
        cratios = list()
        imgs = np.array(imgs.detach().cpu())
        for i in range(imgs.shape[0]):
            img = self.transform.inverse(imgs[i, 0, ...])
            cratios.append(img_util.compute_cratio(img, self.codec))
            if i < 10:
                save_mip_png(f"{i}.png", img)
        return cratios

    def _metric_saturation_margin(self, brain_id):
        """Resolve a per-brain margin, falling back to the loss configuration."""
        provenance = (
            (self.run_config or {}).get("provenance", {}).get(
                "noise_models", {}
            )
        )
        margins = provenance.get("saturation_margins") or {}
        if str(brain_id) in margins:
            return float(margins[str(brain_id)])
        count_cfg = (self.loss_config or {}).get("count") or {}
        return float(count_cfg.get("saturation_margin", 64.0))

    def _log_stratified_metrics(self, report, epoch):
        """Write compact per-stratum and halo summaries to TensorBoard."""
        sections = (
            "by_brain",
            "by_foreground_presence",
            "by_foreground_intensity",
            "by_background_noise",
            "by_saturation",
            "halo_distance_bands",
        )
        for section in sections:
            for label, metrics in report.get(section, {}).items():
                safe_label = str(label).replace("/", "_")
                for name, value in metrics.items():
                    if np.isscalar(value) and np.isfinite(value):
                        self.writer.add_scalar(
                            f"val_strata/{section}/{safe_label}/{name}",
                            value,
                            epoch,
                        )
        for brain_id, bands in report.get("halo_by_brain", {}).items():
            safe_brain_id = str(brain_id).replace("/", "_")
            for band, metrics in bands.items():
                for name, value in metrics.items():
                    if np.isscalar(value) and np.isfinite(value):
                        self.writer.add_scalar(
                            f"val_strata/halo_by_brain/{safe_brain_id}/"
                            f"{band}/{name}",
                            value,
                            epoch,
                        )

    def compute_metrics(self, hat_y, batch, brain_ids):
        """
        Computes per-example neurite-preservation metrics in count space.

        Parameters
        ----------
        hat_y : torch.Tensor
            Model predictions in the normalized transform domain.
        batch : dict[str, torch.Tensor]
            Validation batch containing targets, raw counts, and provenance.
        brain_ids : list[str]
            Indexed brain IDs associated with ``brain_index`` metadata.

        Returns
        -------
        list[dict]
            One structured metric/provenance record per example.
        """
        rows = list()
        preds = np.array(hat_y.detach().cpu())
        targets = np.array(batch["target"].detach().cpu())
        raws = np.array(batch["raw"].detach().cpu())
        masks = np.array(batch["foreground"].detach().cpu())
        brain_indices = np.array(batch["brain_index"].detach().cpu())
        centers = np.array(batch["center"].detach().cpu())
        offsets = np.array(batch["offset"].detach().cpu())
        noise_params = np.array(batch["noise_params"].detach().cpu())
        for i in range(preds.shape[0]):
            pred = self.transform.inverse(preds[i, 0, ...])
            target = self.transform.inverse(targets[i, 0, ...])
            brain_index = int(brain_indices[i])
            brain_id = (
                brain_ids[brain_index]
                if 0 <= brain_index < len(brain_ids)
                else "legacy"
            )
            rows.append(
                evaluate_stratified_example(
                    pred,
                    raws[i, 0, ...],
                    target,
                    masks[i, 0, ...] > 0.5,
                    brain_id=brain_id,
                    center=centers[i],
                    offset=offsets[i],
                    noise_params=noise_params[i],
                    max_count=self.transform.max_count,
                    saturation_margin=self._metric_saturation_margin(brain_id),
                )
            )
        return rows

    def load_pretrained_weights(self, model_path):
        """
        Loads a pretrained model weights from a checkpoint file.

        Parameters
        ----------
        model_path : str
            Path to the checkpoint file containing the saved weights.
        """
        ckpt = torch.load(model_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            checkpoint_model_cfg = ckpt.get("model_config")
            current_model_cfg = getattr(self.model, "config", None)
            if (
                checkpoint_model_cfg is not None
                and current_model_cfg is not None
            ):
                checkpoint_input_channels = checkpoint_model_cfg.get(
                    "in_channels", 1
                )
                current_input_channels = current_model_cfg.get(
                    "in_channels", 1
                )
                if checkpoint_input_channels != current_input_channels:
                    raise ValueError(
                        "resume checkpoint input-channel configuration does "
                        "not match the configured model"
                    )
                normalized_checkpoint_cfg = dict(checkpoint_model_cfg)
                normalized_current_cfg = dict(current_model_cfg)
                normalized_checkpoint_cfg.setdefault("in_channels", 1)
                normalized_current_cfg.setdefault("in_channels", 1)
            else:
                normalized_checkpoint_cfg = checkpoint_model_cfg
                normalized_current_cfg = current_model_cfg
            if (
                normalized_checkpoint_cfg is not None
                and normalized_checkpoint_cfg != normalized_current_cfg
            ):
                raise ValueError(
                    "resume checkpoint model configuration does not match "
                    "the configured model"
                )
            self._resume_transform_cfg = ckpt.get("transform")
            self._resume_run_config = ckpt.get("run_config")
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        self.model.load_state_dict(state_dict)

    def save_config(self, config):
        """
        Writes a run configuration to ``config.json`` in the session directory.

        The training script's hyperparameters are not otherwise persisted, so
        this records them alongside the checkpoints and tensorboard logs to
        make each run reproducible. The Trainer's own hyperparameters are
        merged in so callers cannot forget them.

        Parameters
        ----------
        config : dict
            Run configuration (paths, hyperparameters, transform) assembled by
            the caller. Merged over the Trainer-owned fields.
        """
        record = {
            "batch_size": self.batch_size,
            "device": self.device,
            "max_epochs": self.max_epochs,
            "num_workers": self.num_workers,
            "prefetch": self.prefetch,
            "val_every": self.val_every,
            "fg_weight": getattr(self.criterion, "fg_weight", None),
            "loss_config": self.loss_config,
            "checkpoint_weights": self.checkpoint_weights,
            "checkpoint_selection": self.checkpoint_selection,
            "lr": self.optimizer.param_groups[0]["lr"],
            "model": type(self.model).__name__,
            "model_config": getattr(self.model, "config", None),
        }
        record.update(config)
        self.run_config = record
        util.write_json(os.path.join(self.log_dir, "config.json"), record)

    def save_model(self, epoch, score):
        """
        Saves the current model state to a file.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        score : float
            Checkpoint-selection score for this epoch (lower is better),
            embedded in the filename so checkpoints can be ranked offline.
        """
        date = datetime.today().strftime("%Y%m%d")
        filename_score = (
            score if np.isfinite(score) else float(np.finfo(np.float32).max)
        )
        filename = f"BM4DNet-{date}-{epoch}-{filename_score:.6f}.pth"
        path = os.path.join(self.log_dir, filename)
        torch.save(
            {
                "model": self.model.state_dict(),
                "model_config": getattr(self.model, "config", None),
                "transform": getattr(self.transform, "cfg", None),
                "loss_config": self.loss_config,
                "run_config": self.run_config,
                "provenance": (
                    self.run_config.get("provenance")
                    if self.run_config is not None
                    else None
                ),
                "checkpoint_selection_config": self.checkpoint_selection,
                "checkpoint_selection_result": self.last_checkpoint_selection,
                "checkpoint_score": float(score),
            },
            path,
        )
