""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_rf = False, rf_on_up=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor,use_rf=use_rf)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.use_rf = use_rf
        self.rf_on_up = rf_on_up
        self.set_use_rf(use_rf,rf_on_up)

    def forward(self, x):
        if self.use_rf:
            used_areas = torch.zeros(5).to(self.device)
            x1,used_area = self.inc(x)
            used_areas[0] += used_area
            x2,used_area = self.down1(x1)
            used_areas[1] = used_areas[0]+used_area
            x3,used_area = self.down2(x2)
            used_areas[2] = used_areas[1]+used_area
            x4,used_area = self.down3(x3)
            used_areas[3] = used_areas[2]+used_area
            x5,used_area = self.down4(x4)
            used_areas[4] = used_areas[3]+used_area
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
        if self.use_rf and self.rf_on_up:
            x,used_area = self.up1(x5, x4)
            used_areas[0] += used_area
            prev = used_area
            x,used_area = self.up2(x, x3)
            prev += used_area
            used_areas[1] += prev 
            x,used_area = self.up3(x, x2)
            prev += used_area
            used_areas[2] += prev
            x,used_area = self.up4(x, x1)
            prev += used_area
            used_areas[3] += prev
            used_areas[4] += prev
        else:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        logits = self.outc(x)
        if self.use_rf:
            return logits,used_areas
        return logits
    
    def set_use_rf(self,use_rf,rf_on_up=False):
        self.use_rf = use_rf
        for layer in (self.inc,self.down1,self.down2,self.down3,self.down4):
            layer.set_use_rf(use_rf)
        for layer in (self.up1,self.up2,self.up3,self.up4):
            layer.set_use_rf(use_rf and rf_on_up)