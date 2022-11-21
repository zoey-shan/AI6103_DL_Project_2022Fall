import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_score import dice_loss
import torch.nn as nn
from utils.dice_score import multiclass_dice_coeff, dice_coeff
import numpy as np



def evaluate(net, dataloader, device):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    return_loss = []
    all_loss = []

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        batch_loss = []
        image, mask_true = batch['image'], batch['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true1 = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred1 = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred1, mask_true1, reduce_batch_first=False)
                
                loss = criterion(mask_pred, mask_true) \
                           + dice_loss(F.softmax(mask_pred, dim=1).float(),
                                       F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
                batch_loss.append(loss.item())
                all_loss.append(loss.item())
                
                
            else:
                mask_pred2= F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred2[:, 1:, ...], mask_true1[:, 1:, ...], reduce_batch_first=False)
                loss = criterion(mask_pred, mask_true) \
                           + dice_loss(F.softmax(mask_pred, dim=1).float(),
                                       F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)           
                batch_loss.append(loss.item())
                all_loss.append(loss.item())
    
    net.train()
    batch_loss = np.average(batch_loss)
    return_loss.append(batch_loss)

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score, all_loss
    return dice_score / num_val_batches, all_loss