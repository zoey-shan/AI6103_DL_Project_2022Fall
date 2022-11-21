""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from torch.nn import Dropout


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, dropout_p=0.5):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.dropout1 = Dropout(dropout_p)
        self.down2 = Down(128, 256)
        self.dropout2 = Dropout(dropout_p)
        self.down3 = Down(256, 512)
        self.dropout3 = Dropout(dropout_p)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.dropout4 = Dropout(dropout_p)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.dropout5 = Dropout(dropout_p)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.dropout6 = Dropout(dropout_p)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.dropout7 = Dropout(dropout_p)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.dropout1(x2)
        x3 = self.down2(x2)
        x3 = self.dropout2(x3)
        x4 = self.down3(x3)
        x4 = self.dropout3(x4)
        x5 = self.down4(x4)
        x5 = self.dropout4(x5)
        x = self.up1(x5, x4)
        x = self.dropout5(x)
        x = self.up2(x, x3)
        x = self.dropout6(x)
        x = self.up3(x, x2)
        x = self.dropout7(x)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
