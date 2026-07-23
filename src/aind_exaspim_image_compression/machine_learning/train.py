"""
Created on Fri Jan 3 12:30:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code used to train neural network to denoise images.

"""

from datetime import datetime
from numcodecs import blosc
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import math
import numpy as np
import os
import torch
import torch.optim as optim
from skimage import io

from aind_exaspim_image_compression.machine_learning.unet3d import UNet
from aind_exaspim_image_compression.machine_learning.data_handling import DataLoader
from aind_exaspim_image_compression.machine_learning.losses import (
    SignalPreservingLoss,
)
from aind_exaspim_image_compression.machine_learning.metrics import (
    checkpoint_score,
    evaluate_example,
)
from aind_exaspim_image_compression.utils import img_util, util


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
        use_amp_validation=False,
        checkpoint_weights=None,
        fg_weight=20.0,
        num_workers=None,
        prefetch=2,
        val_every=1000,
        seed=0,
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
            Whether to use CUDA float16 autocast and gradient scaling during
            training. Default is True.
        use_amp_validation : bool, optional
            Whether to use CUDA float16 autocast during validation and
            checkpoint selection. Default is False so validation matches FP32
            production inference.
        val_every : int, optional
            Run validation (and checkpoint selection) every this many gradient
            updates; the final epoch is always validated. Count-space metrics
            are CPU-bound, so a large validation set is only cheap if it is
            not run every epoch. Default is 1000 (validate every 1000 gradient
            updates).
        seed : int, optional
            Seed used to reproducibly shuffle training examples by epoch.
            Default is 0.
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
        self.seed = seed
        self.use_amp = bool(use_amp and device.startswith("cuda"))
        self.use_amp_validation = bool(use_amp_validation and device.startswith("cuda"))

        self.codec = blosc.Blosc(cname="zstd", clevel=6, shuffle=blosc.SHUFFLE)
        self.criterion = SignalPreservingLoss(fg_weight=fg_weight)
        self.checkpoint_weights = checkpoint_weights
        self.model = model.to(device) if model else UNet().to(device)
        self._resume_transform_cfg = None
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.writer = SummaryWriter(log_dir=log_dir)

        # Scale the loss before backward so small float16 gradients do not
        # underflow (and are unscaled before the step). Disabled => no-op, so
        # the same code path is correct with and without AMP.
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

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
        # Create dataloaders
        if self._resume_transform_cfg is not None:
            self.check_transform_cfg(train_dataset, "train")
            self.check_transform_cfg(val_dataset, "val")

        self.transform = train_dataset.transform
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch=self.prefetch,
            shuffle=True,
            seed=self.seed,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch=self.prefetch,
            shuffle=False,
        )

        # Create learning rate scheduler
        total_steps = self.max_epochs * len(train_dataloader)
        expected_validations = math.ceil(total_steps / self.val_every)
        scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        print(f"Total optimizer steps: {total_steps}")
        print(f"Validation interval: every {self.val_every} steps")
        print(f"Expected validations: {expected_validations}")

        # Training loop
        step = 0
        running_loss = 0.0
        running_steps = 0
        self.model.train()
        for epoch in range(self.max_epochs):
            print(f"Starting epoch {epoch} / {self.max_epochs}...")
            train_dataloader.set_epoch(epoch)
            for x, y, fg_mask in train_dataloader:
                # Train
                loss = self.train_step(x, y, fg_mask)
                scheduler.step()

                running_loss += loss
                running_steps += 1
                step += 1

                # Validate (if applicable)
                if step % self.val_every == 0:
                    # Summarize train progress
                    avg_loss = running_loss / running_steps
                    label = f"Step {step}:  train_loss={avg_loss:.5f}, "
                    self.writer.add_scalar("train_loss", avg_loss, step)

                    # Call validation
                    self.validate_and_checkpoint(val_dataloader, step, label)

                    # Reset counters for training session
                    running_loss = 0
                    running_steps = 0
                    self.model.train()

        # Final model validation
        if step % self.val_every != 0:
            self.validate_and_checkpoint(val_dataloader, step, "Final: ")

    def train_step(self, x, y, fg_mask):
        # Forward
        _, loss = self.forward_pass(x, y, fg_mask, use_amp=self.use_amp)

        # Backward
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    def validate_and_checkpoint(self, val_dataloader, step, label):
        # Validate
        loss, cratio, score = self.validate(val_dataloader, step)
        if score is None:
            print(f"{label}: validation skipped (empty validation dataset)")
            return False

        # Save checkpoint and report reuslts
        self.save_model(step, score)
        print(
            f"{label}"
            f"val_loss={loss:.5f}, "
            f"val_cratio={cratio:.5f}, "
            f"val_score={score:.5f}"
        )
        return True

    def validate(self, val_dataloader, step):
        """
        Validates the model over the validation dataset.

        Parameters
        ----------
        val_dataloader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        step : int
            Current number of gradient steps.

        Returns
        -------
        loss : float
            Average loss over the validation dataset.
        cratio : float
            Average compression ratio over the validation dataset.
        score : float or None
            Model score on validation dataset.
        """
        # Skip if there are no validation examples
        if len(val_dataloader.dataset) == 0:
            return float("nan"), float("nan"), None

        # Run model over validation dataset
        losses = list()
        cratios = list()
        metric_rows = list()
        self.model.eval()
        with torch.inference_mode():
            for x, y, raw, fg_mask in val_dataloader:
                # Run model
                y_pred, loss = self.forward_pass(
                    x, y, fg_mask, use_amp=self.use_amp_validation
                )

                # Evaluate result
                losses.append(loss.item())
                cratios.extend(self.compute_cratios(y_pred))
                metric_rows.extend(
                    self.compute_metrics(y_pred, y, raw, fg_mask)
                )

        # Aggregate results
        loss = float(np.mean(losses))
        cratio = float(np.median(cratios))
        agg = {
            k: float(np.mean([row[k] for row in metric_rows]))
            for k in metric_rows[0]
        }
        score = checkpoint_score(agg, cratio, self.checkpoint_weights)

        # Log results
        self.writer.add_scalar("val_loss", loss, step)
        self.writer.add_scalar("val_cratio", cratio, step)
        self.writer.add_scalar("val_score", score, step)
        for name, value in agg.items():
            self.writer.add_scalar(f"val_{name}", value, step)

        return loss, cratio, score

    def forward_pass(self, x, y, fg_mask, use_amp=None):
        """
        Performs a forward pass through the model and computes loss.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, C, D, H, W).
        y : torch.Tensor
            Target tensor with shape (B, C, D, H, W).
        fg_mask : torch.Tensor
            Foreground mask (0/1) with shape (B, C, D, H, W).
        use_amp : bool, optional
            Whether to autocast this forward pass to float16. Defaults to the
            training AMP setting.

        Returns
        -------
        y_pred : torch.Tensor
            Model predictions.
        loss : torch.Tensor
            Computed loss value.
        """
        if use_amp is None:
            use_amp = self.use_amp
        with torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=torch.float16,
            enabled=bool(use_amp),
        ):
            x = x.to(self.device)
            y = y.to(self.device)
            fg_mask = fg_mask.to(self.device)
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y, fg_mask)
            return y_pred, loss

    # --- Helpers ---
    def check_transform_cfg(self, dataset, dataset_name):
        cfg = getattr(dataset.transform, "cfg", None)
        if cfg != self._resume_transform_cfg:
            raise ValueError(
                f"Resume checkpoint transform does not match the "
                f"{dataset_name} dataset transform."
            )

    def compute_cratios(self, imgs):
        cratios = list()
        imgs = np.array(imgs.detach().cpu())
        for i in range(imgs.shape[0]):
            img = self.transform.inverse(imgs[i, 0])
            cratios.append(img_util.compute_cratio(img, self.codec))
            if i < 10:
                save_mip_png(f"{i}.png", img)
        return cratios

    def compute_metrics(self, y_pred, y, raw, fg_mask):
        """
        Computes per-example neurite-preservation metrics in count space.

        Parameters
        ----------
        y_pred : torch.Tensor
            Model predictions in the normalized transform domain.
        y : torch.Tensor
            BM4D targets in the normalized transform domain.
        raw : torch.Tensor
            Raw noisy patches in counts.
        fg_mask : torch.Tensor
            Foreground masks (float 0/1).

        Returns
        -------
        List[dict]
            One metric dictionary per example in the batch.
        """
        rows = list()
        preds = np.array(y_pred.detach().cpu())
        targets = np.array(y.detach().cpu())
        raws = np.array(raw.detach().cpu())
        masks = np.array(fg_mask.detach().cpu())
        for i in range(preds.shape[0]):
            pred = self.transform.inverse(preds[i, 0])
            target = self.transform.inverse(targets[i, 0])
            rows.append(
                evaluate_example(
                    pred, raws[i, 0], target, masks[i, 0] > 0.5
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
                and checkpoint_model_cfg != current_model_cfg
            ):
                raise ValueError(
                    "resume checkpoint model configuration does not match "
                    "the configured model"
                )
            self._resume_transform_cfg = ckpt.get("transform")
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
            "seed": self.seed,
            "use_amp": self.use_amp,
            "use_amp_validation": self.use_amp_validation,
            "fg_weight": getattr(self.criterion, "fg_weight", None),
            "checkpoint_weights": self.checkpoint_weights,
            "lr": self.optimizer.param_groups[0]["lr"],
            "model": type(self.model).__name__,
            "model_config": getattr(self.model, "config", None),
        }
        record.update(config)
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
        score = np.inf if score is None else score
        filename = f"BM4DNet-{date}-{epoch}-{score:.6f}.pth"
        path = os.path.join(self.log_dir, filename)
        torch.save(
            {
                "model": self.model.state_dict(),
                "model_config": getattr(self.model, "config", None),
                "transform": getattr(self.transform, "cfg", None),
            },
            path,
        )


# --- Helpers ---
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
