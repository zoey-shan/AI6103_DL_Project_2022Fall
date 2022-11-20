import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.utils.data import DataLoader, random_split
from utils.data_loading import PhcDataset,SkinCancerDataset
import numpy as np
from unet.unet_model import UNet
import argparse
from utils.dice_score import dice_loss

class PhcDataModule(LightningDataModule):
    def __init__(self,  images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = '', batch_size: int = 32, num_workers: int = 8, **kwargs):
        super().__init__()
        self.phc = PhcDataset(images_dir, masks_dir, scale, mask_suffix)
        self.size = len(self.phc)
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage=None):
        # make assignments here (val/train/test split)
        # called on every GPU
        self.train_dataset,self.val_dataset,self.test_dataset = random_split(self.phc,[int(self.size*0.8),int(self.size*0.1),int(self.size*0.1)],generator=torch.Generator().manual_seed(0))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class SkinCancerDataModule(LightningDataModule):
    def __init__(self,  
            train_x_dir: str, 
            train_y_dir: str, 
            val_x_dir: str,
            val_y_dir: str,
            test_x_dir: str,
            test_y_dir: str,
            scale: float = 1.0, mask_suffix: str = '', batch_size: int = 32, num_workers: int = 8, **kwargs
        ):
        super().__init__()
        # self.phc = (images_dir, masks_dir, scale, mask_suffix)

        self.train_dataset = SkinCancerDataset(train_x_dir, train_y_dir, scale, mask_suffix)
        self.val_dataset = SkinCancerDataset(val_x_dir, val_y_dir, scale, mask_suffix)
        self.test_dataset = SkinCancerDataset(test_x_dir, test_y_dir, scale, mask_suffix)
        # self.size = len(self.phc)
        self.batch_size = batch_size
        self.num_workers = num_workers


    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class UNetLightning(LightningModule):
    def __init__(self, 
        use_rf: bool = False, # Use Receiver-Field Regularization
        rf_on_up: bool = False, # Use Receiver-Field Regularization on Upsampling
        rf_reg_weight: float = 0.1,
        num_classes: int = 2,
        n_channels: int = 1,
        reg_layers: int = 4,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        train_epochs: int = 100,
        **kwargs
        ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = UNet(n_channels=n_channels,n_classes=num_classes, use_rf=use_rf, rf_on_up=rf_on_up)
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.use_rf = use_rf
        self.rf_on_up = rf_on_up
        self.rf_reg_weight = rf_reg_weight
        self.reg_layers = reg_layers
        self.tr_loss = []
        self.tr_acc = []
        self.va_loss = [] 
        self.va_acc = []
        self.tr_temp_loss = []
        self.tr_temp_acc = []
        self.va_temp_loss = []
        self.va_temp_acc = []
        self.tr_dice = []
        self.va_dice = []
        self.tr_temp_dice = []
        self.va_temp_dice = []
        self.rf_reg_loss = []
        self.rf_reg_temp_loss = []


    def forward(self,x):
        # if self.use_rf:
        #     x = torch.tensor(x,requires_grad=True).to(self.device)
        self.model.device = self.device
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        self.model.set_use_rf(self.use_rf,self.rf_on_up)
        # if self.use_rf:
        #     pred_mask, rf_loss = self.forward(x)
        # else: pred_mask = self.forward(x)
        if self.use_rf:
            pred_mask, used_areas = self.forward(x)
        else:
            pred_mask = self.forward(x)
        loss = self.loss_fn(pred_mask, y)
        
        self.log("tr_loss", loss,prog_bar=True)
        dice = dice_loss(F.softmax(pred_mask, dim=1).float(),
                    F.one_hot(y, self.num_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True)
        self.tr_temp_dice.append(dice.item())
        self.log("tr_dice", dice,prog_bar=True)
        loss += dice
        self.tr_temp_loss.append(loss.detach().item())
        if self.use_rf:
            rf_loss = used_areas[self.reg_layers]
            # rf_loss = torch.mean(used_areas[:self.reg_layers])
            freq_reg_loss = self.rf_reg_weight * rf_loss
            self.rf_reg_temp_loss.append(freq_reg_loss.item())
            loss += freq_reg_loss
            # print(rf_loss.requires_grad,used_areas[0].requires_grad)
            self.log("tr_freq_reg", freq_reg_loss,prog_bar=True)
        self.log("tr_total_loss", loss,prog_bar=True)
        self.tr_temp_acc.append(self.accuracy(pred_mask,y))
        self.log("tr_accurcay", self.tr_temp_acc[-1],prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt = torch.optim.Adam(self.parameters(), lr=lr, betas=(b1, b2))
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.train_epochs)

        return [opt],[cosine_scheduler]

    def training_epoch_end(self, training_step_outputs):
        self.tr_loss.append(sum(self.tr_temp_loss)/len(self.tr_temp_loss))
        self.tr_acc.append(sum(self.tr_temp_acc)/len(self.tr_temp_acc))
        self.tr_dice = sum(self.tr_temp_dice)/len(self.tr_temp_dice)
        self.rf_reg_loss.append(sum(self.rf_reg_temp_loss)/len(self.rf_reg_temp_loss))
        self.tr_temp_loss = []
        self.tr_temp_acc = []
        self.tr_temp_dice = []
        self.rf_reg_temp_loss = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        self.model.set_use_rf(False,False)
        pred_mask = self.forward(x)
        loss = self.loss_fn(pred_mask, y)
        dice = dice_loss(F.softmax(pred_mask, dim=1).float(),
                                       F.one_hot(y, self.num_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
        self.va_temp_dice.append(dice.item())
        self.log("va_dice", dice,prog_bar=True)
        loss += dice
        self.va_temp_loss.append(loss.detach().item())
        self.log("va_loss", loss,prog_bar=True)
        self.va_temp_acc.append(self.accuracy(pred_mask,y))
        self.log("va_accurcay", self.va_temp_acc[-1],prog_bar=True)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        self.va_loss.append(sum(self.va_temp_loss)/len(self.va_temp_loss))
        self.va_acc.append(sum(self.va_temp_acc)/len(self.va_temp_acc))
        self.va_dice.append(sum(self.va_temp_dice)/len(self.va_temp_dice))
        self.va_temp_loss = []
        self.va_temp_acc = []
        self.va_temp_dice = []
        print("------validation_epoch_end------")
    

    @staticmethod
    def accuracy(y_hat, y):
        return torch.mean((torch.argmax(y_hat,dim=1) == y)*1.0)

