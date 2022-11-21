import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pytorchtools import EarlyStopping

from utils.data_loading import BasicDataset, CarvanaDataset, PhcDataset, SkinDataset, ValSkinDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_phc = Path('./data/phc')
dir_skin = Path('./data/skinData')
dir_checkpoint = Path('./checkpoints/')


train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []


def train_net(net,
              device,
              epochs: int = 100,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # 1. Create dataset
    try:
        train_set = SkinDataset(dir_skin)
        val_set = ValSkinDataset(dir_skin)
        dataset = PhcDataset(dir_phc)
    except (AssertionError, RuntimeError):
        train_set = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    n_val = len(val_set)
    n_train = len(train_set)
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=False)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='allow')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=0, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0
    
    plot_train_loss = []
    plot_valid_loss = []
    plot_valid_dice = []
    stopping_epoch = 0

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        tr_loss_avg = []
        val_loss_avg = []
        val_acc_avg = []
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                tr_loss_avg.append(loss.item())
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                train_loss_list.append(loss.item())
                pbar.set_postfix(**{'loss (batch)': loss.item()})


                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            if not torch.isinf(value).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not torch.isinf(value.grad).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score, val_loss = evaluate(net, val_loader, device)
                        val_loss_avg.append(val_loss)
                        val_acc_avg.append(val_score.cpu())

                        scheduler.step(val_score)
                        
                        logging.info('Validation Dice score: {}'.format(val_score))
                        # experiment.log({
                        #     'learning rate': optimizer.param_groups[0]['lr'],
                        #     'validation Dice': val_score,
                        #     'validation Loss': val_loss,
                        #     'images': wandb.Image(images[0].cpu()),
                        #     'masks': {
                        #         'true': wandb.Image(true_masks[0].float().cpu()),
                        #         'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                        #     },
                        #     'step': global_step,
                        #     'epoch': epoch,
                        #     **histograms
                        # })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
            experiment.log({
                            'train loss': np.mean(tr_loss_avg),
                            'learning rate': optimizer.param_groups[0]['lr'],
                            # 'validation Dice': val_score,
                            # 'validation Loss': val_loss,
                            'validation Dice': np.mean(val_acc_avg),
                            'validation Loss': np.mean(val_loss_avg),
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })
            
        plot_train_loss.append(np.mean(tr_loss_avg))
        plot_valid_loss.append(np.mean(val_loss_avg))
        plot_valid_dice.append(np.mean(val_acc_avg))
        
        # early stopping check
        epoch_len = len(str(epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                    f'train_loss: {np.mean(tr_loss_avg):.5f} ' +
                    f'valid_loss: {np.mean(val_loss_avg):.5f}' +
                    f'valid_dice: {np.mean(val_acc_avg):.5f}')

        print(print_msg)
        
        early_stopping(np.mean(val_loss_avg), net)

        if early_stopping.early_stop:
            stopping_epoch = epoch
            print("Early stopping", stopping_epoch)
            break
        else:
            print("No Early stop, continous")
    return plot_train_loss, plot_valid_loss, plot_valid_dice, stopping_epoch


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()

def show_plot_epoch(train_loss, valid_loss, valid_dice, stopping_epoch):
    # visualize the loss as the network trained
    fig = plt.figure()
    plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss)+1), valid_loss, label='Validation Loss')
    plt.plot(range(1, len(valid_dice)+1), valid_dice, label='Validation Dice')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1
    print("minposs", minposs)
    print("stopping_epoch", stopping_epoch)
    plt.axvline(minposs, linestyle='--', color='r',
                label='Early Stopping Checkpoint')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    maximum_train = max(train_loss)
    maximum_valid = max(valid_loss)
    maximum_dice = max(valid_dice)
    upper = max(maximum_train, maximum_valid, maximum_dice) + 0.05

    minimum_train = min(train_loss)
    minimum_valid = min(valid_loss)
    minimum_dice = min(valid_dice)
    lower = min(minimum_valid, minimum_train, minimum_dice) - 0.05
    
    plt.ylim(round(lower), round(upper))  # consistent scale
    plt.xlim(0, len(train_loss)+1)  # consistent scale
    
    
    my_y_ticks = np.arange(lower, upper, 0.05)
    my_x_ticks = np.arange(0, len(train_loss)+1, 1)
    plt.yticks(my_y_ticks)
    plt.xticks(my_x_ticks)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('epoch_stop.png', bbox_inches='tight')


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    
    # set up the early stopping features
    bar = 25  # stop the running epoch when continous 25 epoches have no improvement
    early_stopping = EarlyStopping(bar, verbose=True)

    try:
        plot_train_loss, plot_valid_loss, plot_valid_dice, stopping_epoch = train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
        with open("result-constant.txt","w") as f:
            f.write("Train Loss:\n")
            f.write(str(train_loss_list)+"\n")
            f.write("Test Accuracy:\n")
            f.write(str(test_acc_list)+"\n")
        show_plot_epoch(plot_train_loss, plot_valid_loss, plot_valid_dice, stopping_epoch)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
    
# import argparse
# import logging
# import sys
# from pathlib import Path

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import wandb
# from torch import optim
# from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm
# from pytorchtools import EarlyStopping

# from utils.data_loading import BasicDataset, CarvanaDataset, PhcDataset, SkinDataset, ValSkinDataset
# from utils.dice_score import dice_loss
# from evaluate import evaluate
# from unet import UNet

# import numpy as np
# import matplotlib.pyplot as plt
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
# dir_phc = Path('./data/phc')
# dir_skin = Path('./data/skinData')
# dir_checkpoint = Path('./checkpoints/')

# train_loss_list = []
# train_acc_list = []
# val_loss_list = []
# val_acc_list = []

# epoch_plot_train_loss = []
# epoch_plot_val_loss = []
# epoch_plot_val_score = []

# every_train_loss_list = []

# def train_net(net,
#               device,
#               epochs: int = 50,
#               batch_size: int = 1,
#               learning_rate: float = 1e-5,
#               val_percent: float = 0.1,
#               save_checkpoint: bool = True,
#               img_scale: float = 0.5,
#               amp: bool = False):
#     # 1. Create dataset
#     try:
#         #dataset = SkinDataset(dir_skin)
#         #val_dataset = ValSkinDataset(dir_skin)
#         dataset = PhcDataset(dir_phc)
#     except (AssertionError, RuntimeError):
#         dataset = BasicDataset(dir_img, dir_mask, img_scale)

#     # 2. Split into train / validation partitions
#     n_val = int(len(dataset) * val_percent)
#     n_train = len(dataset) - n_val
#     train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

#     # 3. Create data loaders
#     loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=False)
#     train_loader = DataLoader(train_set, shuffle=True, **loader_args)
#     val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

#     # (Initialize logging)
#     experiment = wandb.init(project='U-Net', resume='allow', anonymous='allow')
#     experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
#                                   val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
#                                   amp=amp))

#     logging.info(f'''Starting training:
#         Epochs:          {epochs}
#         Batch size:      {batch_size}
#         Learning rate:   {learning_rate}
#         Training size:   {n_train}
#         Validation size: {n_val}
#         Checkpoints:     {save_checkpoint}
#         Device:          {device.type}
#         Images scaling:  {img_scale}
#         Mixed Precision: {amp}
#     ''')

#     # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
#     optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=0, momentum=0.9)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
#     grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
#     criterion = nn.CrossEntropyLoss()
#     global_step = 0
#     stopping_epoch = 0

#     # 5. Begin training
#     for epoch in range(1, epochs+1):
#         epoch_train_loss = []
#         epoch_val_loss = []
#         epoch_val_score = []
#         net.train()
#         epoch_loss = 0
#         with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
#             for batch in train_loader:
#                 images = batch['image']
#                 true_masks = batch['mask']

#                 assert images.shape[1] == net.n_channels, \
#                     f'Network has been defined with {net.n_channels} input channels, ' \
#                     f'but loaded images have {images.shape[1]} channels. Please check that ' \
#                     'the images are loaded correctly.'

#                 images = images.to(device=device, dtype=torch.float32)
#                 true_masks = true_masks.to(device=device, dtype=torch.long)

#                 with torch.cuda.amp.autocast(enabled=amp):
#                     masks_pred = net(images)
#                     loss = criterion(masks_pred, true_masks) \
#                            + dice_loss(F.softmax(masks_pred, dim=1).float(),
#                                        F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
#                                        multiclass=True)

#                 #set_to_none=True
#                 optimizer.zero_grad(set_to_none=True)
#                 grad_scaler.scale(loss).backward()
#                 grad_scaler.step(optimizer)
#                 grad_scaler.update()

#                 pbar.update(images.shape[0])
#                 global_step += 1
#                 epoch_loss += loss.item()
#                 experiment.log({
#                     'train loss': loss.item(),
#                     'step': global_step,
#                     'epoch': epoch
#                 })
#                 epoch_train_loss.append(loss.item())
#                 every_train_loss_list.append(loss.item())
#                 pbar.set_postfix(**{'loss (batch)': loss.item()})


#                 # Evaluation round
#                 division_step = (n_train // (10 * batch_size))
#                 if division_step > 0:
#                     if global_step % division_step == 0:
#                         train_loss_list.append(loss.item())
#                         histograms = {}
#                         for tag, value in net.named_parameters():
#                             tag = tag.replace('/', '.')
#                             if not torch.isinf(value).any():
#                                 histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
#                             if not torch.isinf(value.grad).any():
#                                 histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

#                         val_score, val_loss = evaluate(net, val_loader, device)
#                         scheduler.step(val_score)
#                         val_acc_list.append(val_score.item())
#                         epoch_val_score.append(val_score.item())
#                         epoch_val_loss.append(val_loss)
#                         val_loss_list.append(val_loss)
                        

#                         logging.info('Validation Dice score: {}'.format(val_score))
#                         experiment.log({
#                             'learning rate': optimizer.param_groups[0]['lr'],
#                             'validation Dice': val_score,
#                             'images': wandb.Image(images[0].cpu()),
#                             'masks': {
#                                 'true': wandb.Image(true_masks[0].float().cpu()),
#                                 'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
#                             },
#                             'step': global_step,
#                             'epoch': epoch,
#                             **histograms
#                         })
        
#         # epoch_train_loss_final = epoch_train_loss[-1]
#         # epoch_val_score_final = epoch_val_score[-1]
#         # epoch_val_loss_final = epoch_val_loss[-1]
#         epoch_train_loss_final = np.average(epoch_train_loss)
#         epoch_val_score_final = np.average(epoch_val_score)
#         epoch_val_loss_final = np.average(epoch_val_loss)
        
#         print("epoch_val_loss", epoch_val_loss)
        
#         epoch_plot_train_loss.append(epoch_train_loss_final)
#         epoch_plot_val_loss.append(epoch_val_loss_final)
#         epoch_plot_val_score.append(epoch_val_score_final)
        
#         print("epoch_train_loss_final", epoch, epoch_train_loss_final)
#         print("epoch_val_loss_final", epoch, epoch_val_loss_final)
#         print("epoch_val_score_final", epoch, epoch_val_score_final)
        
#         # early stopping check
#         epoch_len = len(str(epochs))
#         print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
#                     f'train_loss: {epoch_train_loss_final:.5f} ' +
#                     f'valid_loss: {epoch_val_loss_final:.5f}' +
#                     f'valid_dice: {epoch_val_score_final:.5f}')

#         print(print_msg)
        
#         early_stopping(epoch_val_loss_final, net)

#         if early_stopping.early_stop:
#             print("Early stopping")
#             stopping_epoch = epoch
#             break
#         else:
#             print("No Early stop, continous")
        
#         if save_checkpoint:
#             Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
#             torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
#             logging.info(f'Checkpoint {epoch} saved!')
#     return stopping_epoch, every_train_loss_list, train_acc_list, train_loss_list, val_acc_list, val_loss_list, epoch_plot_train_loss, epoch_plot_val_loss, epoch_plot_val_score

# def get_args():
#     parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
#     parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
#     parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
#     parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
#                         help='Learning rate', dest='lr')
#     parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
#     parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
#     parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
#                         help='Percent of the data that is used as validation (0-100)')
#     parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
#     parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
#     parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

#     return parser.parse_args()

# def show_plot_epoch(train_loss, valid_loss):
#     # visualize the loss as the network trained
#     fig = plt.figure(figsize=(10, 8))
#     plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss')
#     plt.plot(range(1, len(valid_loss)+1), valid_loss, label='Validation Loss')

#     # find position of lowest validation loss
#     minposs = valid_loss.index(min(valid_loss))+1
#     print("minposs", minposs)
#     print("stopping_epoch", stopping_epoch)
#     plt.axvline(minposs, linestyle='--', color='r',
#                 label='Early Stopping Checkpoint')

#     plt.xlabel('epoch')
#     plt.ylabel('loss')
    
#     maximum_train = max(train_loss)
#     maximum_valid = max(valid_loss)
#     upper = max(maximum_train, maximum_valid) + 0.05

#     minimum_train = min(train_loss)
#     minimum_valid = min(valid_loss)
#     lower = min(minimum_valid, minimum_train) - 0.05
    
#     plt.ylim(round(lower), round(upper))  # consistent scale
#     plt.xlim(0, len(train_loss)+1)  # consistent scale
    
    
#     my_y_ticks = np.arange(lower, upper, 0.05)
#     plt.yticks(my_y_ticks)

#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#     fig.savefig('epoch_stop.png', bbox_inches='tight')
    
# def show_plot_valid_dice(valid_dice):
#     # visualize the loss as the network trained
#     fig = plt.figure(figsize=(10, 8))
#     # plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss')
#     plt.plot(range(1, len(valid_dice)+1), valid_dice, label='Validation Dice')

#     # find position of lowest validation loss
#     maxposs = valid_dice.index(min(valid_dice))+1
#     plt.axvline(maxposs, linestyle='--', color='r',
#                 label='Early Stopping Checkpoint')

#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     upper = max(valid_dice) + 0.05
#     lower = min(valid_dice) - 0.05
#     plt.ylim(round(lower), round(upper))  # consistent scale
#     plt.xlim(0, len(valid_dice)+1)  # consistent scale
    
    
#     my_y_ticks = np.arange(lower, upper, 0.05)
#     plt.yticks(my_y_ticks)

#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#     fig.savefig('epoch_dice_stop.png', bbox_inches='tight')
    
# def show_plot_step(train_loss, valid_loss, every_train_loss_list):
#     # visualize the loss as the network trained
#     fig = plt.figure(figsize=(10, 8))
#     plt.plot(range(1, len(train_loss)+1), train_loss, label='Training Loss')
#     plt.plot(range(1, len(valid_loss)+1), valid_loss, label='Validation Loss')

#     # find position of lowest validation loss
#     minposs = valid_loss.index(min(valid_loss))+1
#     plt.axvline(minposs, linestyle='--', color='r',
#                 label='Early Stopping Checkpoint')

#     plt.xlabel('step')
#     plt.ylabel('loss')
#     maximum_train = max(train_loss)
#     maximum_valid = max(valid_loss)
#     upper = max(maximum_train, maximum_valid) + 0.05

#     minimum_train = min(train_loss)
#     minimum_valid = min(valid_loss)
#     lower = min(minimum_valid, minimum_train) - 0.05
    
#     plt.ylim(round(lower), round(upper))  # consistent scale
#     plt.xlim(0, len(valid_loss)+1)  # consistent scale
    
#     # steps = len(every_train_loss_list) // len(train_loss)
#     # my_x_ticks = np.arange(0, len(every_train_loss_list)+1, steps)
#     # my_y_ticks = np.arange(lower, upper, 0.03)
#     # plt.xticks(my_x_ticks)
#     # plt.yticks(my_y_ticks)
    
    
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#     fig.savefig('step_stop.png', bbox_inches='tight')
    
    
    
# if __name__ == '__main__':
#     args = get_args()

#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Using device {device}')

#     # Change here to adapt to your data
#     # n_channels=3 for RGB images
#     # n_classes is the number of probabilities you want to get per pixel
#     net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

#     logging.info(f'Network:\n'
#                  f'\t{net.n_channels} input channels\n'
#                  f'\t{net.n_classes} output channels (classes)\n'
#                  f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

#     if args.load:
#         net.load_state_dict(torch.load(args.load, map_location=device))
#         logging.info(f'Model loaded from {args.load}')

#     net.to(device=device)
    
#     # set up the early stopping features
#     bar = 7  # stop the running epoch when continous 7 epoches have no improvement
#     early_stopping = EarlyStopping(bar, verbose=True)
    
#     try:
#         stopping_epoch, every_train_loss_list, train_acc_list, train_loss_list, val_acc_list, val_loss_list, epoch_plot_train_loss, epoch_plot_val_loss, epoch_plot_val_score = train_net(net=net,
#                   epochs=args.epochs,
#                   batch_size=args.batch_size,
#                   learning_rate=args.lr,
#                   device=device,
#                   img_scale=args.scale,
#                   val_percent=args.val / 100,
#                   amp=args.amp)
#         with open("result-constant.txt","w") as f:
#             f.write("Train Loss:\n")
#             f.write(str(train_loss_list)+"\n")
#             f.write("Test Accuracy:\n")
#             f.write(str(val_acc_list)+"\n")
            
#         show_plot_epoch(epoch_plot_train_loss, epoch_plot_val_loss)
#         print("epoch_plot_train_loss",epoch_plot_train_loss)
#         print("epoch_plot_val_loss", epoch_plot_val_loss)
#         show_plot_step(train_loss_list, val_loss_list, every_train_loss_list)
#         show_plot_valid_dice(epoch_plot_val_score)
#         print("epoch_plot_val_score", epoch_plot_val_score)
#     except KeyboardInterrupt:
#         torch.save(net.state_dict(), 'INTERRUPTED.pth')
#         logging.info('Saved interrupt')
#         raise

# import argparse
# import logging
# import sys
# from pathlib import Path

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import wandb
# from torch import optim
# from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm

# from utils.data_loading import BasicDataset, CarvanaDataset, PhcDataset, SkinDataset
# from utils.dice_score import dice_loss
# from evaluate import evaluate
# from unet import UNet
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
# dir_phc = Path('./data/phc')
# dir_checkpoint = Path('./checkpoints/')
# dir_skin = Path('./data/skinData')

# train_loss_list = []
# train_acc_list = []
# test_loss_list = []
# test_acc_list = []


# def train_net(net,
#               device,
#               epochs: int = 50,
#               batch_size: int = 1,
#               learning_rate: float = 1e-5,
#               val_percent: float = 0.1,
#               save_checkpoint: bool = True,
#               img_scale: float = 0.5,
#               amp: bool = False):
    # # 1. Create dataset
    # try:
    #     dataset = SkinDataset(dir_skin)
    #     val_dataset = SkinDataset(dir_skin)
    # except (AssertionError, RuntimeError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # # 2. Split into train / validation partitions
    # # n_val = int(len(dataset) * val_percent)
    # # n_train = len(dataset) - n_val
    # # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    # n_train = len(dataset)
    # n_val = len(val_dataset)

    # # 3. Create data loaders
    # loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=False)
    # train_loader = DataLoader(dataset, shuffle=True, **loader_args)
    # val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    # # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='allow')
    # experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #                               val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
    #                               amp=amp))

    # logging.info(f'''Starting training:
    #     Epochs:          {epochs}
    #     Batch size:      {batch_size}
    #     Learning rate:   {learning_rate}
    #     Training size:   {n_train}
    #     Validation size: {n_val}
    #     Checkpoints:     {save_checkpoint}
    #     Device:          {device.type}
    #     Images scaling:  {img_scale}
    #     Mixed Precision: {amp}
    # ''')

    # # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=0, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss()
    # global_step = 0

    # # 5. Begin training
    # for epoch in range(1, epochs+1):
    #     net.train()
    #     epoch_loss = 0
    #     with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
    #         for batch in train_loader:
    #             images = batch['image']
    #             true_masks = batch['mask']

    #             assert images.shape[1] == net.n_channels, \
    #                 f'Network has been defined with {net.n_channels} input channels, ' \
    #                 f'but loaded images have {images.shape[1]} channels. Please check that ' \
    #                 'the images are loaded correctly.'

                # images = images.to(device=device, dtype=torch.float32)
                # true_masks = true_masks.to(device=device, dtype=torch.long)

                # with torch.cuda.amp.autocast(enabled=amp):
                #     masks_pred = net(images)
                #     loss = criterion(masks_pred, true_masks) \
                #            + dice_loss(F.softmax(masks_pred, dim=1).float(),
                #                        F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                #                        multiclass=True)

                # #set_to_none=True
                # optimizer.zero_grad(set_to_none=True)
                # grad_scaler.scale(loss).backward()
                # grad_scaler.step(optimizer)
                # grad_scaler.update()

                # pbar.update(images.shape[0])
                # global_step += 1
                # epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                # train_loss_list.append(loss.item())
                # pbar.set_postfix(**{'loss (batch)': loss.item()})


#                 # Evaluation round
#                 division_step = (n_train // (10 * batch_size))
#                 if division_step > 0:
#                     if global_step % division_step == 0:
#                         histograms = {}
#                         for tag, value in net.named_parameters():
#                             tag = tag.replace('/', '.')
#                             if not torch.isinf(value).any():
#                                 histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
#                             if not torch.isinf(value.grad).any():
#                                 histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

#                         val_score = evaluate(net, val_loader, device)
#                         scheduler.step(val_score)
#                         test_acc_list.append(val_score)

#                         logging.info('Validation Dice score: {}'.format(val_score))
#                         experiment.log({
#                             'learning rate': optimizer.param_groups[0]['lr'],
#                             'validation Dice': val_score,
#                             'images': wandb.Image(images[0].cpu()),
#                             'masks': {
#                                 'true': wandb.Image(true_masks[0].float().cpu()),
#                                 'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
#                             },
#                             'step': global_step,
#                             'epoch': epoch,
#                             **histograms
#                         })

#         if save_checkpoint:
#             Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
#             torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
#             logging.info(f'Checkpoint {epoch} saved!')


# def get_args():
#     parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
#     parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
#     parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
#     parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
#                         help='Learning rate', dest='lr')
#     parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
#     parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
#     parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
#                         help='Percent of the data that is used as validation (0-100)')
#     parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
#     parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
#     parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

#     return parser.parse_args()


# if __name__ == '__main__':
#     args = get_args()

#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Using device {device}')

#     # Change here to adapt to your data
#     # n_channels=3 for RGB images
#     # n_classes is the number of probabilities you want to get per pixel
#     net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

#     logging.info(f'Network:\n'
#                  f'\t{net.n_channels} input channels\n'
#                  f'\t{net.n_classes} output channels (classes)\n'
#                  f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

#     if args.load:
#         net.load_state_dict(torch.load(args.load, map_location=device))
#         logging.info(f'Model loaded from {args.load}')

#     net.to(device=device)
#     try:
#         train_net(net=net,
#                   epochs=args.epochs,
#                   batch_size=args.batch_size,
#                   learning_rate=args.lr,
#                   device=device,
#                   img_scale=args.scale,
#                   val_percent=args.val / 100,
#                   amp=args.amp)
#         with open("result-constant.txt","w") as f:
#             f.write("Train Loss:\n")
#             f.write(str(train_loss_list)+"\n")
#             f.write("Test Accuracy:\n")
#             f.write(str(test_acc_list)+"\n")
#     except KeyboardInterrupt:
#         torch.save(net.state_dict(), 'INTERRUPTED.pth')
#         logging.info('Saved interrupt')
#         raise

# import argparse
# import logging
# import sys
# from pathlib import Path

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import wandbß
# from torch import optim
# from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm

# import utils.data_loading 
# from utils.dice_score import dice_loss
# from evaluate import evaluate
# from unet import UNet

# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
# dir_checkpoint = Path('./checkpoints/')
# phc_path = Path('./data/phcData')
# skin_path = Path('./data/skinData')


# def train_net(net,
#               device,
#               epochs: int = 50,
#               batch_size: int = 1,
#               learning_rate: float = 1e-5,
#               val_percent: float = 0.1,
#               save_checkpoint: bool = True,
#               img_scale: float = 0.5,
#               amp: bool = False):
#     # 1. Create dataset
#     try:
#         # dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
#         dataset = utils.data_loading.SkinDataset(skin_path, img_scale)
#         print("data loaded")
#     except (AssertionError, RuntimeError):
#         dataset = utils.data_loading.BasicDataset(dir_img, dir_mask, img_scale)

#     # 2. Split into train / validation partitions
#     n_val = int(len(dataset) * val_percent)
#     n_train = len(dataset) - n_val
#     train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

#     # 3. Create data loaders
#     loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=False)
#     train_loader = DataLoader(train_set, shuffle=True, **loader_args)
#     val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

#     # (Initialize logging)
#     experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
#     experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
#                                   val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
#                                   amp=amp))

#     logging.info(f'''Starting training:
#         Epochs:          {epochs}
#         Batch size:      {batch_size}
#         Learning rate:   {learning_rate}
#         Training size:   {n_train}
#         Validation size: {n_val}
#         Checkpoints:     {save_checkpoint}
#         Device:          {device.type}
#         Images scaling:  {img_scale}
#         Mixed Precision: {amp}
#     ''')

#     # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
#     optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=0, momentum=0.9)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
#     grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
#     criterion = nn.CrossEntropyLoss()
#     global_step = 0

#     # 5. Begin training
#     for epoch in range(1, epochs+1):
#         net.train()
#         epoch_loss = 0
#         with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
#             for batch in train_loader:
#                 images = batch['image']
#                 true_masks = batch['mask']

#                 assert images.shape[1] == net.n_channels, \
#                     f'Network has been defined with {net.n_channels} input channels, ' \
#                     f'but loaded images have {images.shape[1]} channels. Please check that ' \
#                     'the images are loaded correctly.'

#                 images = images.to(device=device, dtype=torch.float32)
#                 true_masks = true_masks.to(device=device, dtype=torch.long)

#                 with torch.cuda.amp.autocast(enabled=amp):
#                     masks_pred = net(images)
#                     loss = criterion(masks_pred, true_masks) \
#                            + dice_loss(F.softmax(masks_pred, dim=1).float(),
#                                        F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
#                                        multiclass=True)

#                 #set_to_none=True
#                 optimizer.zero_grad()
#                 grad_scaler.scale(loss).backward()
#                 grad_scaler.step(optimizer)
#                 grad_scaler.update()

#                 pbar.update(images.shape[0])
#                 global_step += 1
#                 epoch_loss += loss.item()
#                 experiment.log({
#                     'train loss': loss.item(),
#                     'step': global_step,
#                     'epoch': epoch
#                 })
#                 pbar.set_postfix(**{'loss (batch)': loss.item()})

#                 # Evaluation round
#                 division_step = (n_train // (10 * batch_size))
#                 if division_step > 0:
#                     if global_step % division_step == 0:
#                         histograms = {}
#                         for tag, value in net.named_parameters():
#                             tag = tag.replace('/', '.')
#                             if not torch.isinf(value).any():
#                                 histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
#                             if not torch.isinf(value.grad).any():
#                                 histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

#                         val_score = evaluate(net, val_loader, device)
#                         scheduler.step(val_score)
                        
#                         # validation loss
#                         for batch_vali in val_loader:
#                             images = batch_vali['image']
#                             true_masks = batch_vali['mask']

#                             assert images.shape[1] == net.n_channels, \
#                                 f'Network has been defined with {net.n_channels} input channels, ' \
#                                 f'but loaded images have {images.shape[1]} channels. Please check that ' \
#                                 'the images are loaded correctly.'

#                             images = images.to(device=device, dtype=torch.float32)
#                             true_masks = true_masks.to(device=device, dtype=torch.long)
#                             with torch.cuda.amp.autocast(enabled=amp):
#                                 masks_pred = net(images)
#                                 loss = criterion(masks_pred, true_masks) \
#                                         + dice_loss(F.softmax(masks_pred, dim=1).float(),
#                                         F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
#                                         multiclass=True)
                                        
#                             experiment.log({
#                                 'validation loss': loss.item(),
#                                 'step': global_step,
#                                 'epoch': epoch
#                             })

#                         logging.info('Validation Dice score: {}'.format(val_score))
#                         experiment.log({
#                             'learning rate': optimizer.param_groups[0]['lr'],
#                             'validation Dice': val_score,
#                             'images': wandb.Image(images[0].cpu()),
#                             'masks': {
#                                 'true': wandb.Image(true_masks[0].float().cpu()),
#                                 'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
#                             },
#                             'step': global_step,
#                             'epoch': epoch,
#                             **histograms
#                         })
                        
#         # validation loss
#         # for batch in val_loader:
#         #     images = batch['image']
#         #     true_masks = batch['mask']

#         #     assert images.shape[1] == net.n_channels, \
#         #         f'Network has been defined with {net.n_channels} input channels, ' \
#         #         f'but loaded images have {images.shape[1]} channels. Please check that ' \
#         #         'the images are loaded correctly.'

#         #     images = images.to(device=device, dtype=torch.float32)
#         #     true_masks = true_masks.to(device=device, dtype=torch.long)
#         #     with torch.cuda.amp.autocast(enabled=amp):
#         #             masks_pred = net(images)
#         #             loss = criterion(masks_pred, true_masks) \
#         #                    + dice_loss(F.softmax(masks_pred, dim=1).float(),
#         #                                F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
#         #                                multiclass=True)
                           
#         #     experiment.log({
#         #             'validation loss': loss.item(),
#         #             'step': global_step,
#         #             'epoch': epoch
#         #         })
            
#         if save_checkpoint:
#             Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
#             torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
#             logging.info(f'Checkpoint {epoch} saved!')


# def get_args():
#     parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
#     parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
#     parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
#     parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
#                         help='Learning rate', dest='lr')
#     parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
#     parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
#     parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
#                         help='Percent of the data that is used as validation (0-100)')
#     parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
#     parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
#     parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

#     return parser.parse_args()


# if __name__ == '__main__':
#     args = get_args()

#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Using device {device}')

#     # Change here to adapt to your data
#     # n_channels=3 for RGB images
#     # n_classes is the number of probabilities you want to get per pixel
#     net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

#     logging.info(f'Network:\n'
#                  f'\t{net.n_channels} input channels\n'
#                  f'\t{net.n_classes} output channels (classes)\n'
#                  f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

#     if args.load:
#         net.load_state_dict(torch.load(args.load, map_location=device))
#         logging.info(f'Model loaded from {args.load}')

#     net.to(device=device)
#     try:
#         train_net(net=net,
#                   epochs=args.epochs,
#                   batch_size=args.batch_size,
#                   learning_rate=args.lr,
#                   device=device,
#                   img_scale=args.scale,
#                   val_percent=args.val / 100,
#                   amp=args.amp)
#     except KeyboardInterrupt:
#         torch.save(net.state_dict(), 'INTERRUPTED.pth')
#         logging.info('Saved interrupt')
#         raise
