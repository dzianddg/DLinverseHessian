import torch
import torch.nn as nn
import torch.nn.functional as F

class R2Block(nn.Module):
    """Input => ConvBlock1 => ConvBlock2 = x1
       Input => Shortcut = x2
       x = ReLU(x1 + x2)"""

    def __init__(self, in_channels, out_channels, mid_channels=None, t=2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.t = t
        self.rec_conv_block = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.input_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        # Create shortcut
        xs = self.shortcut(x)

        # Perform Recurrent UNet
        x = self.input_conv(x)
        x0 = self.rec_conv_block(x)
        for _ in range(self.t):
            x0 = self.rec_conv_block(x + x0)
        x = x0
        
        x0 = self.rec_conv_block(x)
        for _ in range(self.t):
            x0 = self.rec_conv_block(x + x0)
        x = x0

        # Add shortcut layer
        x = torch.add(x, xs)
        x = self.relu(x)
        return x

class Down_R2UNet(nn.Module):
    """Downscaling with maxpool then double conv block for residual UNet"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            R2Block(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x

class Up_R2UNet(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = R2Block(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = R2Block(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x