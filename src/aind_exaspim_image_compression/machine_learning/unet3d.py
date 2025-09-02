"""
Created on Fri Aug 14 15:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that implements a 3D U-Net.

"""

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

    def __init__(self, width_multiplier=1, trilinear=True, use_relu=True):
        """
        Instantiates a UNet object.

        Parameters
        ----------
        width_multiplier : float, optional
            Factor that scales the number of channels in each layer. Default
            is 1.
        trilinear : bool, optional
            If True, use trilinear interpolation for upsampling in decoder
            blocks; otherwise, use transposed convolutions. Default is True.
        use_relu : bool, optional
            If True, use ReLU activations in DoubleConv blocks; otherwise,
            use LeakyReLU. Default is True.
        """
        # Call parent class
        super(UNet, self).__init__()

        # Initializations
        _channels = (32, 64, 128, 256, 512)
        factor = 2 if trilinear else 1

        # Instance attributes
        self.channels = [int(c * width_multiplier) for c in _channels]
        self.trilinear = trilinear

        # Contracting layers
        self.inc = DoubleConv(1, self.channels[0], use_relu=use_relu)
        self.down1 = Down(self.channels[0], self.channels[1])
        self.down2 = Down(self.channels[1], self.channels[2])
        self.down3 = Down(self.channels[2], self.channels[3])
        self.down4 = Down(self.channels[3], self.channels[4] // factor)

        # Expanding layers
        self.up1 = Up(self.channels[4], self.channels[3] // factor, trilinear)
        self.up2 = Up(self.channels[3], self.channels[2] // factor, trilinear)
        self.up3 = Up(self.channels[2], self.channels[1] // factor, trilinear)
        self.up4 = Up(self.channels[1], self.channels[0], trilinear)
        self.outc = OutConv(self.channels[0], 1)

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
        # Contracting layers
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Expanding layers
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """
    A module that consists of two consecutive 3D convolutional layers, each
    followed by batch normalization and a nonlinear activation.

    Attributes
    ----------
    double_conv : nn.Sequential
        Sequential module containing two convolutions, batch norms, and
        activations.
    """

    def __init__(
        self, in_channels, out_channels, mid_channels=None, use_relu=True
    ):
        """
        Instantiates a DoubleConv object.

        Parameters
        ----------
        in_channels : int
            Number of input channels to this module.
        out_channels : int
            Number of output channels produced by this module.
        mid_channels : int, optional
            Number of channels in the intermediate convolution. Default is
            None.
        use_relu : bool, optional
            If True, use ReLU activations; otherwise use LeakyReLU. Default
            is True.
        """
        # Call parent class
        super().__init__()

        # Check whether to set custom mid channel dimension
        if not mid_channels:
            mid_channels = out_channels

        # Set nonlinear activation
        if use_relu:
            activation = nn.ReLU(inplace=True)
        else:
            activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        # Instance attributes
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            activation,
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            activation
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

    Attributes
    ----------
    maxpool_conv : nn.Sequential
        Sequential module containing a MaxPool3d layer followed by a
        DoubleConv block.
    """

    def __init__(self, in_channels, out_channels):
        """
        Instantiates a Down object.

        Parameters
        ----------
        in_channels : int
            Number of input channels to this module.
        out_channels : int
            Number of output channels produced by this module.
        """
        # Call parent class
        super().__init__()

        # Instance attributes
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2), DoubleConv(in_channels, out_channels)
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

    def __init__(self, in_channels, out_channels, trilinear=True):
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
        if trilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="trilinear", align_corners=True
            )
            self.conv = DoubleConv(
                in_channels, out_channels, mid_channels=in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward pass of the upsampling block in a 3D U-Net.

        Parameters
        ----------
        x1 : torch.Tensor
            Input tensor from the previous decoder layer with shape
            (B, C1, D, H1, W1).
        x2 : torch.Tensor
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
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


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
