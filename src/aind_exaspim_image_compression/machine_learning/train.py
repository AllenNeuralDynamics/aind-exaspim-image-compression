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
import tifffile
import torch
import torch.nn as nn
import torch.optim as optim

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
        checkpoint_weights=None,
        fg_weight=20.0,
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

        self.codec = blosc.Blosc(cname="zstd", clevel=5, shuffle=blosc.SHUFFLE)
        self.criterion = SignalPreservingLoss(fg_weight=fg_weight)
        self.checkpoint_weights = checkpoint_weights
        self.best_score = np.inf
        self.model = model.to(device) if model else UNet().to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=25)
        self.writer = SummaryWriter(log_dir=log_dir)

        if use_amp:
            self.autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            self.autocast = nullcontext()

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
        self.transform = train_dataset.transform
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size
        )
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Main
        self.best_score = np.inf
        for epoch in range(self.max_epochs):
            # Train-Validate
            train_loss = self.train_step(train_dataloader, epoch)
            val_loss, val_cratio, is_best = self.validate_step(
                val_dataloader, epoch
            )

            # Report results
            suffix = " - New Best!" if is_best else ""
            s = f"Epoch {epoch}:  train_loss={train_loss},  val_loss={val_loss}, val_cratio={val_cratio}" + suffix
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
        for x, y, fg_mask in train_dataloader:
            # Forward pass
            hat_y, loss = self.forward_pass(x, y, fg_mask)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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
        metric_rows = list()
        with torch.no_grad():
            self.model.eval()
            for x, y, raw, fg_mask in val_dataloader:
                # Run model
                hat_y, loss = self.forward_pass(x, y, fg_mask)

                # Evaluate result
                losses.append(loss.detach().cpu())
                cratios.extend(self.compute_cratios(hat_y))
                metric_rows.extend(
                    self.compute_metrics(hat_y, y, raw, fg_mask)
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
        self.writer.add_scalar("val_loss", loss, epoch)
        self.writer.add_scalar("val_cratio", cratio, epoch)
        self.writer.add_scalar("val_score", score, epoch)
        for name, value in agg.items():
            self.writer.add_scalar(f"val_{name}", value, epoch)

        # Check if current model is best so far (lower score is better)
        is_best = score < self.best_score
        if is_best:
            self.best_score = score
            self.save_model(epoch)
        return loss, cratio, is_best

    def forward_pass(self, x, y, fg_mask):
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

        Returns
        -------
        hat_y : torch.Tensor
            Model predictions.
        loss : torch.Tensor
            Computed loss value.
        """
        with self.autocast:
            x = x.to(self.device)
            y = y.to(self.device)
            fg_mask = fg_mask.to(self.device)
            hat_y = self.model(x)
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
                tifffile.imwrite(f"{i}.tiff", img)
        return cratios

    def compute_metrics(self, hat_y, y, raw, fg_mask):
        """
        Computes per-example neurite-preservation metrics in count space.

        Parameters
        ----------
        hat_y : torch.Tensor
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
        preds = np.array(hat_y.detach().cpu())
        targets = np.array(y.detach().cpu())
        raws = np.array(raw.detach().cpu())
        masks = np.array(fg_mask.detach().cpu())
        for i in range(preds.shape[0]):
            pred = self.transform.inverse(preds[i, 0, ...])
            target = self.transform.inverse(targets[i, 0, ...])
            rows.append(
                evaluate_example(
                    pred, raws[i, 0, ...], target, masks[i, 0, ...] > 0.5
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
            ckpt = ckpt["model"]
        self.model.load_state_dict(ckpt)

    def save_model(self, epoch):
        """
        Saves the current model state to a file.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        """
        date = datetime.today().strftime("%Y%m%d")
        filename = f"BM4DNet-{date}-{epoch}-{self.best_score:.6f}.pth"
        path = os.path.join(self.log_dir, filename)
        torch.save(
            {
                "model": self.model.state_dict(),
                "transform": getattr(self.transform, "cfg", None),
            },
            path,
        )
