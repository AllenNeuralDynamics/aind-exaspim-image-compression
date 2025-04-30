"""
Created on Fri Jan 3 12:30:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code used to train neural network to classify somas proposals.

"""

from datetime import datetime
from numcodecs import blosc
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tifffile import imwrite
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from aind_exaspim_image_compression.machine_learning.unet3d import UNet
from aind_exaspim_image_compression.machine_learning.data_handling import (
    TrainBM4DDataLoader,
    ValidateBM4DDataLoader,
)
from aind_exaspim_image_compression.utils import img_util, util


class Trainer:
    def __init__(
        self,
        output_dir,
        batch_size=8,
        lr=1e-3,
        max_epochs=200,
    ):
        # Initializations
        exp_name = "session-" + datetime.today().strftime("%Y%m%d_%H%M")
        log_dir = os.path.join(output_dir, exp_name)
        util.mkdir(log_dir)

        # Instance attributes
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.log_dir = log_dir

        self.codec = blosc.Blosc(cname="zstd", clevel=5, shuffle=blosc.SHUFFLE)
        self.criterion = nn.L1Loss()
        self.model = UNet().to("cuda")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=25)
        self.writer = SummaryWriter(log_dir=log_dir)

    def run(self, train_dataset, val_dataset, n_upds=25):
        # Initializations
        print("Experiment:", os.path.basename(os.path.normpath(self.log_dir)))
        train_dataloader = TrainBM4DDataLoader(
            train_dataset, batch_size=self.batch_size, n_upds=n_upds
        )
        val_dataloader = ValidateBM4DDataLoader(
            val_dataset, batch_size=self.batch_size,
        )

        # Main
        self.best_l1 = np.inf
        for epoch in range(self.max_epochs):
            # Updates
            train_loss = self.train_step(train_dataloader, epoch)
            val_loss, val_cratio, new_best = self.validate_model(val_dataloader, epoch)
            if new_best:
                print(f"Epoch {epoch}:  train_loss={train_loss},  val_loss={val_loss}, val_cratio={val_cratio} - New Best!")
            else:
                print(f"Epoch {epoch}:  train_loss={train_loss},  val_loss={val_loss}, val_cratio={val_cratio}")
                

            # Step scheduler
            self.scheduler.step()

    def train_step(self, train_dataloader, epoch):
        losses = list()
        self.model.train()
        for x_i, y_i, _ in train_dataloader:
            # Forward pass
            x_i, y_i = x_i.to("cuda"), y_i.to("cuda")
            hat_y_i = self.model(x_i)
            loss = self.criterion(hat_y_i, y_i)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Store loss for tensorboard
            losses.append(float(loss.detach().cpu()))
        self.writer.add_scalar("train_loss", np.mean(losses), epoch)
        return np.mean(losses)

    def validate_model(self, val_dataloader, epoch):
        losses = list()
        cratios = list()
        self.model.eval()
        with torch.no_grad():
            for x, y, mn_mx in val_dataloader:
                # Run model
                x, y = x.to("cuda"), y.to("cuda")
                hat_y = self.model(x)
                loss = self.criterion(hat_y, y)                

                # Evalute result
                cratios.extend(self.compute_cratios(hat_y, mn_mx))
                losses.append(loss.detach().cpu())

        # Log results
        loss, cratio = np.mean(losses), np.mean(cratios)
        self.writer.add_scalar("val_loss", loss, epoch)
        self.writer.add_scalar("val_cratio", cratio, epoch)
        if loss < self.best_l1:
            self.save_model(epoch)
            self.best_l1 = loss
            return loss, cratio, True
        else:
            return loss, cratio, False

    def compute_cratios(self, imgs, mn_mx):
        cratios = list()
        imgs = np.array(imgs.detach().cpu())
        for i in range(imgs.shape[0]):
            mn, mx = tuple(mn_mx[i, :])
            img = (imgs[i, 0, ...] * mx + mn).astype(np.uint16)
            cratios.append(
                img_util.compute_cratio(img, self.codec)
            )
        return cratios

    def save_model(self, epoch):
        date = datetime.today().strftime("%Y%m%d")
        filename = f"GraphNeuralNet-{date}-{epoch}-{round(self.best_l1, 4)}.pth"
        path = os.path.join(self.log_dir, filename)
        torch.save(self.model.state_dict(), path)
