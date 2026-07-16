
"""
Created on Fri Aug 14 15:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that implements a 3D U-Net.

"""

from math import gcd
from numbers import Real

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    3D U-Net architecture for 3D image data, suitable for tasks such as
    denoising or segmentation.

    Attributes
    ----------
    channels : List[int]
        Number of channels in each layer after applying "width_multiplier".
    trilinear : bool
        Flag indicating whether trilinear upsampling is used.
    inc : DoubleConv
        Initial convolution block.
    down1, down2, down3, down4 : Down
        Downsampling blocks in the encoder path.
    up1, up2, up3, up4 : Up
        Upsampling blocks in the decoder path.
    outc : OutConv
        Final 1x1x1 convolution mapping features to the output channel.
    """

    def __init__(
        self,
        width_multiplier=1,
        trilinear=True,
        residual=True,
        maxblurpool=False,
        remove_top_skip=False,
    ):
        """
        Instantiates a UNet object.

        Parameters
        ----------
        width_multiplier : int, optional
            Positive integer factor that scales the number of channels in each
            layer. Default is 1.
        trilinear : bool, optional
            If True, use trilinear interpolation for upsampling in decoder
            blocks; otherwise, use transposed convolutions. Default is True.
        residual : bool, optional
            If True, the network predicts a residual added to the input, so it
            learns to "remove noise" rather than reconstruct the full signal.
            Default is True.
        """
        # Call parent class
        super().__init__()

        if (
            isinstance(width_multiplier, Real)
            or width_multiplier < 1
            or not float(width_multiplier).is_integer()
        ):
            raise ValueError("width_multiplier must be a positive integer")

        # Initializations
        base_channels = (32, 64, 128, 256, 512)
        factor = 2 if trilinear else 1

        # Instance attributes
        self.width_multiplier = int(width_multiplier)
        self.channels = [
            c * self.width_multiplier
            for c in base_channels
        ]

        self.trilinear = trilinear
        self.residual = residual
        self.maxblurpool = maxblurpool
        self.remove_top_skip = remove_top_skip

        # Encoder
        self.inc = DoubleConv(1, self.channels[0])

        self.down1 = Down(
            self.channels[0],
            self.channels[1],
            maxblurpool=maxblurpool,
        )

        self.down2 = Down(
            self.channels[1],
            self.channels[2],
            maxblurpool=maxblurpool,
        )

        self.down3 = Down(
            self.channels[2],
            self.channels[3],
            maxblurpool=maxblurpool,
        )

        self.down4 = Down(
            self.channels[3],
            self.channels[4] // factor,
            maxblurpool=maxblurpool,
        )

        # Decoder
        self.up1 = Up(
            self.channels[4],
            self.channels[3] // factor,
            trilinear=trilinear,
            use_skip=True,
        )

        self.up2 = Up(
            self.channels[3],
            self.channels[2] // factor,
            trilinear=trilinear,
            use_skip=True,
        )

        self.up3 = Up(
            self.channels[2],
            self.channels[1] // factor,
            trilinear=trilinear,
            use_skip=True,
        )

        self.up4 = Up(
            self.channels[1],
            self.channels[0],
            trilinear=trilinear,
            use_skip=not remove_top_skip,
        )

        self.outc = OutConv(self.channels[0], 1)

    @property
    def config(self):
        """Constructor arguments needed to recreate this model."""
        return {
            "width_multiplier": self.width_multiplier,
            "trilinear": self.trilinear,
            "residual": self.residual,
            "maxblurpool": self.maxblurpool,
            "remove_top_skip": self.remove_top_skip,
        }

    def forward(self, x):
        """
        Forward pass of the 3D U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, 1, D, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (B, 1, D, H, W), representing the
            denoised image.
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        d = self.up1(x5, x4)
        d = self.up2(d, x3)
        d = self.up3(d, x2)

        if self.remove_top_skip:
            d = self.up4(d)
        else:
            d = self.up4(d, x1)

        logits = self.outc(d)

        # Residual denoising: predict the correction added to the input
        if self.residual:
            return x + logits

        return logits


class DoubleConv(nn.Module):
    """
    A module that consists of two consecutive 3D convolutional layers, each
    followed by group normalization and a nonlinear activation.

    Attributes
    ----------
    double_conv : nn.Sequential
        Sequential module containing two convolutions, group norms, and
        activations.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, mid_channels=None
    ):
        """
        Instantiates a DoubleConv object.

        Parameters
        ----------
        in_channels : int
            Number of input channels to this module.
        out_channels : int
            Number of output channels produced by this module.
        kernel_size : int, optional
            Size of kernel used in convolutional layers. Default is 3.
        mid_channels : int, optional
            Number of channels in the intermediate convolution. Default is
            None.
        """
        # Call parent class
        super().__init__()

        # Check whether to set custom mid channel dimension
        if not mid_channels:
            mid_channels = out_channels

        # Instance attributes
        self.double_conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=1,
            ),
            nn.GroupNorm(gcd(8, mid_channels), mid_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=1,
            ),
            nn.GroupNorm(gcd(8, out_channels), out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        """
        Forward pass of the double convolution module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor after double convolution.
        """
        return self.double_conv(x)


class Down(nn.Module):
    """
    A downsampling module for a 3D U-Net.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        maxblurpool=False,
    ):
        """
        Instantiates a Down object.

        Parameters
        ----------
        in_channels : int
            Number of input channels to this module.
        out_channels : int
            Number of output channels produced by this module.
        maxblurpool : bool, optional
            True if max-blur pooling should be used to downsample. Default is
            False.
        """
        # Call parent class
        super().__init__()

        # Initializations
        if maxblurpool:
            downsample = MaxBlurPool3D(in_channels)
        else:
            downsample = nn.MaxPool3d(2)

        # Instance attributes
        self.maxpool_conv = nn.Sequential(
            downsample,
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        """
        Forward pass of the downsampling block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor after max pooling and double convolution.
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    An upsampling block for a 3D U-Net that performs spatial upscaling
    followed by a double convolution.

    Attributes
    ----------
    up : nn.Module
        Upsampling layer (either nn.Upsample or nn.ConvTranspose3d).
    conv : DoubleConv
        Double convolution block applied after concatenating the skip
        connection.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        trilinear=True,
        use_skip=True,
    ):
        """
        Instantiates an Up object.

        Parameters
        ----------
        in_channels : int
            Number of input channels to this module.
        out_channels : int
            Number of output channels produced by this module.
        trilinear : bool, optional
            Indication of whether to use nn.Upsample or nn.ConvTranspose3d.
            Default is True, meaning that nn.Upsample is used.
        """
        # Call parent class
        super().__init__()

        # Instance attributes
        self.use_skip = use_skip

        if trilinear:
            self.up = nn.Upsample(
                scale_factor=2,
                mode="trilinear",
                align_corners=True,
            )
        else:
            self.up = nn.ConvTranspose3d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )

        conv_in_channels = (
            in_channels if use_skip else in_channels // 2
        )

        self.conv = DoubleConv(
            conv_in_channels,
            out_channels,
        )

    def forward(self, x1, x2=None):
        """
        Forward pass of the upsampling block in a 3D U-Net.

        Parameters
        ----------
        x1 : torch.Tensor
            Input tensor from the previous decoder layer with shape
            (B, C1, D, H1, W1).
        x2 : torch.Tensor, optional
            Skip connection tensor from the encoder path with shape
            (B, C2, D, H2, W2).

        Returns
        -------
        torch.Tensor
            Output tensor after upsampling, concatenation with the skip
            connection, and double convolution. The output shape is
            (B, out_channels, D, H2, W2).
        """
        x1 = self.up(x1)

        if self.use_skip:
            diff_z = x2.size(2) - x1.size(2)
            diff_y = x2.size(3) - x1.size(3)
            diff_x = x2.size(4) - x1.size(4)

            x1 = F.pad(
                x1,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                    diff_z // 2,
                    diff_z - diff_z // 2,
                ],
            )

            x = torch.cat([x2, x1], dim=1)

        else:
            x = x1

        return self.conv(x)


class MaxBlurPool3D(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.pool = nn.MaxPool3d(2, stride=1)

        kernel = torch.tensor([1., 2., 1.])
        kernel = (
            kernel[:, None, None]
            * kernel[None, :, None]
            * kernel[None, None, :]
        )
        kernel /= kernel.sum()

        self.register_buffer(
            "kernel",
            kernel[None, None].repeat(channels, 1, 1, 1, 1),
        )
        self.channels = channels

    def forward(self, x):
        x = self.pool(x)
        x = F.conv3d(
            x,
            self.kernel,
            stride=2,
            padding=1,
            groups=self.channels,
        )
        return x


class OutConv(nn.Module):
    """
    Final output convolution layer for a 3D U-Net.

    Attributes
    ----------
    conv : nn.Conv3d
        1x1x1 convolution that maps the feature channels to the output
        channels.
    """

    def __init__(self, in_channels, out_channels):
        """
        Instantiates an OutConv object.

        Parameters
        ----------
        in_channels : int
            Number of input channels to this module.
        out_channels : int
            Number of output channels produced by this module.
        """
        # Call parent class
        super(OutConv, self).__init__()

        # Instance attributes
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the output convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor from the last decoder layer with shape
            (B, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor after 1x1x1 convolution, with shape
            (B, 1, D, H, W).
        """
        return self.conv(x)
