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

import numpy as np
import os
import tifffile
import torch
import torch.nn as nn
import torch.optim as optim

from aind_exaspim_image_compression.machine_learning.unet3d import UNet
from aind_exaspim_image_compression.machine_learning.data_handling import DataLoader
from aind_exaspim_image_compression.utils import img_util, util


class Trainer:

    def __init__(
        self,
        output_dir,
        batch_size=8,
        device="cuda:0",
        lr=1e-3,
        max_epochs=200,
    ):
        """
        Instantiates a Trainer object.

        Parameters
        ----------
        output_dir : str
            Directory that model checkpoints and tensorboard are written to.
        batch_size : int, optional
            Number of samples per batch during training. Default is 32.
        device : str, optional
            GPU device that model is trained on. Default is "cuda:0".
        lr : float, optional
            Learning rate. Default is 1e-3.
        max_epochs : int, optional
            Maximum number of training epochs. Default is 200.
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
        self.criterion = nn.L1Loss()
        self.model = UNet(use_relu=False).to("cuda")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=25)
        self.writer = SummaryWriter(log_dir=log_dir)

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
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size
        )
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Main
        self.best_l1 = np.inf
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
        for x, y, _ in train_dataloader:
            # Forward pass
            hat_y, loss = self.forward_pass(x, y)

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
        tuple
            A tuple containing the following:
            - float: Average loss over the validation dataset.
            - float: Average compression ratio over the validation dataset.
            - bool: Indication of whether the model is the best so far.
        """
        losses = list()
        cratios = list()
        with torch.no_grad():
            self.model.eval()
            for x, y, mn_mx in val_dataloader:
                # Run model
                hat_y, loss = self.forward_pass(x, y)

                # Evalute result
                cratios.extend(self.compute_cratios(hat_y, mn_mx))
                losses.append(loss.detach().cpu())

        # Log results
        loss, cratio = np.mean(losses), np.mean(cratios)
        self.writer.add_scalar("val_loss", loss, epoch)
        self.writer.add_scalar("val_cratio", cratio, epoch)

        # Check if current model is best so far
        if loss < self.best_l1:
            self.best_l1 = loss
            self.save_model(epoch)
            return loss, cratio, True
        else:
            return loss, cratio, False

    def forward_pass(self, x, y):
        """
        Performs a forward pass through the model and computes loss.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, C, D, H, W).
        y : torch.Tensor
            Ground truth labels with shape (B, C, D, H, W).

        Returns
        -------
        hat_y : torch.Tensor
            Model predictions.
        loss : torch.Tensor
            Computed loss value.
        """
        x = x.to("cuda")
        y = y.to("cuda")
        hat_y = self.model(x)
        loss = self.criterion(hat_y, y)
        return hat_y, loss

    # --- Helpers ---
    def compute_cratios(self, imgs, mn_mx):
        cratios = list()
        imgs = np.array(imgs.detach().cpu())
        for i in range(imgs.shape[0]):
            mn, mx = tuple(mn_mx[i, :])
            img = imgs[i, 0, ...] * (mx - mn) + mn
            cratios.append(img_util.compute_cratio(img, self.codec))
            if i < 10:
                tifffile.imwrite(f"{i}.tiff", img)
        return cratios

    def load_pretrained_weights(self, model_path):
        """
        Loads a pretrained model weights from a checkpoint file.
    
        Parameters
        ----------
        model_path : str
            Path to the checkpoint file containing the saved weights.
        """
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

    def save_model(self, epoch):
        """
        Saves the current model state to a file.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        """
        date = datetime.today().strftime("%Y%m%d")
        filename = f"BM4DNet-{date}-{epoch}-{self.best_l1:.4f}.pth"
        path = os.path.join(self.log_dir, filename)
        torch.save(self.model.state_dict(), path)
