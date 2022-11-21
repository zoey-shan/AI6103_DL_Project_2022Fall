import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, PhcDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet

# import the augmentation classes
from utils.allAugmentation_replace import *

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system') 
import matplotlib.pyplot as plt

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')
phc_path = Path('./data/phc/')


# the number of training epochs is set to be 50
# the valid plot for this training is the EpochPlot showing the training and validation loss of each epoch
def train_net(net,
              device,
              epochs: int = 50,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):

    # define the augmentation transform 
    train_transform = DoubleCompose([
        DoubleToTensor(),
        DoubleHorizontalFlip(),
        #DoubleVerticalFlip(),
        DoubleGuassianBlur(),
        DoubleCropAndPad()
    ])

    # 1. Create dataset
    try:
        dataset = PhcDataset(phc_path, img_scale, transform=train_transform)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
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

    # save the loss/score for each epoch
    epochValidScore = []
    epochValidLoss = []
    epochTrainLoss = []

    # save all the train/validation loss and validation score
    all_valid_loss = []
    all_train_loss = []

    # 5. Begin training
    for epoch in range(1, epochs+1):
        # record all the training/validation loss for each epoch
        trainLoss = []
        valLoss = []

        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:

            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                print("images and true mask shape")
                print(true_masks.shape)
                print(images.shape)

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    print(masks_pred)

                    print('predict size')
                    print(images.shape)
                    print(masks_pred.shape)
                    print(true_masks.shape)
                    # only batches of spatial targets supported (3D tensors) but got targets of dimension: 4
                    # torch.Size([1, 1, 520, 696])
                    # torch.Size([1, 2, 520, 696])
                    # torch.Size([1, 1, 520, 696])

                    # # determine the class type with the highest prob
                    # masks_pred = torch.argmax(masks_pred, dim=1)
                    # print("after argmax")
                    # print(masks_pred.shape)

                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # record all training loss
                all_train_loss.append(loss.item())
                trainLoss.append(loss.item())

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

                        val_score, loss_val = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

                        # record the validation score
                        for i in loss_val:
                            all_valid_loss.append(i)
                            valLoss.append(i)
                            
                            

        # store the last training loss and validation score for each epoch
        epochTrainLoss.append(np.average(trainLoss))
        epochValidLoss.append(np.average(valLoss))

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

    print("self define for all train and val loss")
    print(all_train_loss)
    print(all_valid_loss)

    print("epoch train/val loss")
    print(epochTrainLoss)
    print(epochValidLoss)
    
    print("epoch loss/score size")
    print(len(epochTrainLoss), len(epochValidLoss))
    print("batch loss/score size")
    print(len(all_train_loss), len(all_valid_loss))

    # plot the performance curve for each epochs
    show_plot_epoch(epochTrainLoss, epochValidLoss)
    # plot the training loss and validation loss for each step
    show_plot_lossBatch(all_train_loss, all_valid_loss)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
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


# def show_plot_batch(trainLoss, batchScore):
#     # visualize the loss as the network trained
#     plt.subplot(1, 2, 1)
#     plt.plot(range(1,len(trainLoss)+1), trainLoss)
#     plt.xlabel('Steps/Batches')
#     plt.ylabel('Training loss')
#     plt.grid(True)
#     plt.tight_layout()

#     plt.subplot(1,2,2)
#     plt.plot(range(1,len(batchScore)+1), batchScore)
#     plt.xlabel('Steps/Batches')
#     plt.ylabel('Validation score')
#     plt.grid(True)
#     plt.tight_layout()
    
#     plt.show()
#     plt.savefig('Batch-loss/score_plot.png', bbox_inches='tight')

def show_plot_epoch(train_loss, valid_loss):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
    plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    upper = max(train_loss)
    plt.ylim(0, upper) # consistent scale
    plt.xlim(0, len(train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('50_Epoch_trainVal_loss_plot.png', bbox_inches='tight') 


def show_plot_lossBatch(all_train_loss, all_valid_loss):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(all_train_loss)+1), all_train_loss, label='Training Loss')
    plt.plot(range(1,len(all_valid_loss)+1), all_valid_loss,label='Validation Loss')

    plt.xlabel('steps')
    plt.ylabel('loss')
    upper = max(all_train_loss)
    plt.ylim(0, upper) # consistent scale
    plt.xlim(0, len(all_train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('50_Batch_trainVal_loss_plot.png', bbox_inches='tight')

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
