from .attention_r2_unet_parts import *


class AttentionR2UNet(nn.Module):
    """Attention-based Recurrent Residual U-Net (Attention R2U-Net).

    This model is an encoder–decoder convolutional neural network that combines:
    - Recurrent Residual convolutional blocks (R2 blocks),
    - U-Net style skip connections,
    - Attention Gates to selectively emphasize informative encoder features
      during decoding.

    The architecture is commonly used for dense prediction tasks such as
    segmentation, inversion, or image-to-image regression where spatial
    localization and contextual awareness are critical.

    Args:
        n_channels (int): Number of input channels (e.g., 1 for grayscale,
            2 for stacked inputs).
        n_classes (int): Number of output channels or classes.
        hidden_channels (int): Base number of feature channels used throughout
            the network.
        bilinear (bool, optional): If True, use bilinear upsampling instead of
            transposed convolutions in the decoder. Defaults to False.
    """

    def __init__(self, n_channels, n_classes, hidden_channels, bilinear=False):
        super(AttentionR2UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = R2Block(n_channels, hidden_channels)
        self.down1 = Down_R2UNet(hidden_channels, hidden_channels * 2)
        self.down2 = Down_R2UNet(hidden_channels * 2, hidden_channels * 4)
        self.down3 = Down_R2UNet(hidden_channels * 4, hidden_channels * 8)

        factor = 2 if bilinear else 1
        self.down4 = Down_R2UNet(
            hidden_channels * 8, hidden_channels * 16 // factor
        )

        # Attention gates + Decoder
        self.ag1 = AttentionGate(hidden_channels * 16, hidden_channels * 8)
        self.up1 = Up_R2UNet(
            hidden_channels * 16,
            hidden_channels * 8 // factor,
            bilinear,
        )

        self.ag2 = AttentionGate(hidden_channels * 8 // factor, hidden_channels * 4)
        self.up2 = Up_R2UNet(
            hidden_channels * 8,
            hidden_channels * 4 // factor,
            bilinear,
        )

        self.ag3 = AttentionGate(hidden_channels * 4 // factor, hidden_channels * 2)
        self.up3 = Up_R2UNet(
            hidden_channels * 4,
            hidden_channels * 2 // factor,
            bilinear,
        )

        self.ag4 = AttentionGate(hidden_channels * 2 // factor, hidden_channels)
        self.up4 = Up_R2UNet(
            hidden_channels * 2,
            hidden_channels,
            bilinear,
        )

        # Output layer
        self.outc = OutConv(hidden_channels, n_classes)

    def forward(self, x):
        """Forward pass of the Attention R2U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape
                `(batch_size, n_channels, height, width)`.

        Returns:
            torch.Tensor: Output tensor of shape
            `(batch_size, n_classes, height, width)`.
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with attention gates
        ag1 = self.ag1(x5, x4)
        x = self.up1(x5, ag1)

        ag2 = self.ag2(x, x3)
        x = self.up2(x, ag2)

        ag3 = self.ag3(x, x2)
        x = self.up3(x, ag3)

        ag4 = self.ag4(x, x1)
        x = self.up4(x, ag4)

        # Output
        x = self.outc(x)
        return x

    def use_checkpointing(self):
        """Enable gradient checkpointing to reduce memory usage.

        This method wraps major network blocks with PyTorch checkpointing,
        trading additional computation for lower GPU memory consumption.
        Useful when training large models or using high-resolution inputs.
        """
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
