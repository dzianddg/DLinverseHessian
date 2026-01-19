import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class R2Block(nn.Module):
    """Recurrent Residual Convolutional Block (R2 Block).

    This block implements a recurrent residual structure used in R2U-Net.
    The block applies recurrent convolutions multiple times and adds a
    shortcut connection from the input to the output.

    Architecture:
        Input
          ├── 1×1 Conv (shortcut)
          └── 1×1 Conv → Recurrent Conv Block (t times)
        Output = ReLU(recurrent_output + shortcut)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (int, optional): Number of intermediate feature channels.
            If None, defaults to `out_channels`.
        t (int, optional): Number of recurrent iterations. Defaults to 2.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, t=2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.t = t

        self.rec_conv_block = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.input_conv = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, bias=False
        )
        self.shortcut = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass of the R2 block.

        Args:
            x (torch.Tensor): Input tensor of shape
                `(batch_size, in_channels, height, width)`.

        Returns:
            torch.Tensor: Output tensor of shape
            `(batch_size, out_channels, height, width)`.
        """
        # Shortcut connection
        xs = self.shortcut(x)

        # Recurrent convolution
        x = self.input_conv(x)
        x0 = self.rec_conv_block(x)
        for _ in range(self.t):
            x0 = self.rec_conv_block(x + x0)
        x = x0

        # Second recurrent stage
        x0 = self.rec_conv_block(x)
        for _ in range(self.t):
            x0 = self.rec_conv_block(x + x0)
        x = x0

        # Add shortcut and apply activation
        x = torch.add(x, xs)
        x = self.relu(x)
        return x


class AttentionGate(nn.Module):
    """Attention Gate for U-Net skip connections.

    The attention gate suppresses irrelevant encoder features and highlights
    salient regions by conditioning skip connections on decoder features.

    Args:
        in_channels_g (int): Number of channels in the gating signal
            (decoder feature map).
        in_channels_x (int): Number of channels in the skip connection
            (encoder feature map).
    """

    def __init__(self, in_channels_g, in_channels_x):
        super().__init__()
        self.conv_x = nn.Conv2d(
            in_channels_x,
            in_channels_x,
            kernel_size=1,
            stride=2,
            bias=False,
        )
        self.conv_g = nn.Conv2d(
            in_channels_g,
            in_channels_x,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels_x, 1, kernel_size=1, padding="same", bias=True
        )
        self.sigmoid = nn.Sigmoid()
        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

    def forward(self, g, x):
        """Forward pass of the attention gate.

        Args:
            g (torch.Tensor): Gating signal from the decoder.
            x (torch.Tensor): Skip connection feature map from the encoder.

        Returns:
            torch.Tensor: Attention-weighted skip connection.
        """
        x_skip = x

        x = self.conv_x(x)
        g = self.conv_g(g)

        # Ensure spatial dimensions match
        if x.shape[2:] != g.shape[2:]:
            x = F.interpolate(
                x, size=g.shape[2:], mode="bilinear", align_corners=True
            )

        x1 = torch.add(x, g)
        x1 = self.relu(x1)
        x1 = self.conv(x1)
        x1 = self.sigmoid(x1)

        x1 = self.up(x1)

        if x_skip.shape[2:] != x1.shape[2:]:
            x1 = F.interpolate(
                x1,
                size=x_skip.shape[2:],
                mode="bilinear",
                align_corners=True,
            )

        x = torch.mul(x_skip, x1)
        return x


class Down_R2UNet(nn.Module):
    """Downsampling block for R2U-Net.

    This block performs spatial downsampling using max pooling,
    followed by an R2 convolutional block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            R2Block(in_channels, out_channels),
        )

    def forward(self, x):
        """Forward pass of the downsampling block.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Downsampled feature map.
        """
        x = self.maxpool_conv(x)
        return x


class Up_R2UNet(nn.Module):
    """Upsampling block for R2U-Net.

    This block upsamples the input feature map and concatenates it with
    the corresponding skip connection before applying an R2 block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bilinear (bool, optional): If True, use bilinear upsampling.
            If False, use transposed convolution. Defaults to True.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = R2Block(
                in_channels, out_channels, in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = R2Block(in_channels, out_channels)

    def forward(self, x1, x2):
        """Forward pass of the upsampling block.

        Args:
            x1 (torch.Tensor): Decoder feature map to be upsampled.
            x2 (torch.Tensor): Encoder feature map for skip connection.

        Returns:
            torch.Tensor: Fused feature map after upsampling and convolution.
        """
        x1 = self.up(x1)

        # Compute padding to match spatial dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
            ],
        )

        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """Output convolution layer.

    This layer maps feature channels to the desired number of output channels
    using a 1×1 convolution.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )

    def forward(self, x):
        """Forward pass of the output layer.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Output feature map.
        """
        x = self.conv(x)
        return x
