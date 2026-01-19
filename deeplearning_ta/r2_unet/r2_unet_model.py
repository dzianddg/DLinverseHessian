from .r2_unet_parts import *

class R2UNet(nn.Module):
    def __init__(self, n_channels, n_classes, hidden_channels, bilinear=False):
        super(R2UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (R2Block(n_channels, hidden_channels))
        self.down1 = (Down_R2UNet(hidden_channels, hidden_channels*2))
        self.down2 = (Down_R2UNet(hidden_channels*2, hidden_channels*4))
        self.down3 = (Down_R2UNet(hidden_channels*4, hidden_channels*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down_R2UNet(hidden_channels*8, hidden_channels*16 // factor))
        self.up1 = (Up_R2UNet(hidden_channels*16, hidden_channels*8 // factor, bilinear))
        self.up2 = (Up_R2UNet(hidden_channels*8, hidden_channels*4 // factor, bilinear))
        self.up3 = (Up_R2UNet(hidden_channels*4, hidden_channels*2 // factor, bilinear))
        self.up4 = (Up_R2UNet(hidden_channels*2, hidden_channels, bilinear))
        self.outc = (OutConv(hidden_channels, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
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