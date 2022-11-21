# reference from https://github.com/hayashimasa/UNet-PyTorch
# self-modify the vertical flip and elastic transform

"""
Reference from:
Model Trainer
author: Masahiro Hayashi
This script defines custom image transformations that simultaneously transform
both images and segmentation masks.
"""
import torchvision.transforms.functional as TF
import torchvision.transforms as T

import tensorflow as tf

# from torchvision.transforms import Compose
from torchvision import transforms
import torch
from torch.utils.tensorboard import SummaryWriter
import random
from random import randint
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import torch.nn.functional as F



class DoubleToTensor:
    """Apply horizontal flips to both image and segmentation mask."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask, weight=None):
        if weight is None:
            return TF.to_tensor(image), TF.to_tensor(mask)
        weight = weight.view(1, *weight.shape)
        return TF.to_tensor(image), TF.to_tensor(mask), weight

    def __repr__(self):
        return self.__class__.__name__ + '()'

class DoubleHorizontalFlip:
    """Apply horizontal flips to both image and segmentation mask."""

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, image, mask, weight=None):
        #p = random.random()
        p = 0.1
        if p < self.p:
            #print("horizontal flip")
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if weight is None:
            return image, mask
        elif p > self.p:
            weight = TF.hflip(weight)
        return image, mask, weight

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'

class DoubleVerticalFlip:
    """Apply vertical flips to both image and segmentation mask."""

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, image, mask, weight=None):
        #p = random.random()
        p = 0.1
        if p < self.p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        if weight is None:
            return image, mask
        elif p > self.p:
            weight = TF.hflip(weight)
        return image, mask, weight

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'


class DoubleCropAndPad:
    """
    Apply croping to both the image and mask
    Padding is also applied to return both the images in the required shape
    """

    def __init__(self, p=0.1, height = 256, width = 192):
        self.p = p
        self.height = height
        self.width = width

    def __call__(self, image, mask):
        #p = random.random()
        p = 0.1
        if p < self.p:
            
            toImage = T.ToPILImage()
            params = transforms.RandomResizedCrop.get_params(image, (0.7, 1.0), (1,1))
            padImage= TF.resized_crop(toImage(image), *params, (self.height, self.width))
            padMask = TF.resized_crop(toImage(mask), *params, (self.height, self.width))

            return TF.to_tensor(padImage), TF.to_tensor(padMask)

        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'

class DoubleElasticTransform:
    """Based on implimentation on
    https://gist.github.com/erniejunior/601cdf56d2b424757de5"""

    def __init__(self, alpha=250, sigma=10, p=0.1, seed=None, randinit=True):
        if not seed:
            seed = random.randint(1, 100)
        self.random_state = np.random.RandomState(seed)
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self.randinit = randinit


    def __call__(self, image, mask, weight=None):

        #p = random.random()
        p = 0.1
        if p < self.p:
            if self.randinit:
                seed = random.randint(1, 100)
                self.random_state = np.random.RandomState(seed)
                self.alpha = random.uniform(100, 300)
                self.sigma = random.uniform(10, 15)

            # elastic transform for image
            dim = image.shape

            dx = self.alpha * gaussian_filter(
                (self.random_state.rand(*dim) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0
            )
            dy = self.alpha * gaussian_filter(
                (self.random_state.rand(*dim) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0
            )
            dz = np.zeros_like(dx)

            image = image.view(*dim[:]).numpy()
            x, y, z = np.meshgrid(np.arange(dim[1]), np.arange(dim[0]), np.arange(dim[2]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))           
            image = map_coordinates(image, indices, order=1)            
            image = image.reshape(dim)
            image = torch.Tensor(image)

            # elastic transform for image
            dim = mask.shape

            dx = self.alpha * gaussian_filter(
                (self.random_state.rand(*dim[1:]) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0
            )
            dy = self.alpha * gaussian_filter(
                (self.random_state.rand(*dim[1:]) * 2 - 1),
                self.sigma,
                mode="constant",
                cval=0
            )
            mask = mask.view(*dim[1:]).numpy()
            x, y = np.meshgrid(np.arange(dim[2]), np.arange(dim[1]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))           
            mask = map_coordinates(mask, indices, order=1)            
            mask = mask.reshape(dim)
            mask = torch.Tensor(mask)

            print(image.shape)
            print(mask.shape)

            if weight is None:
                return image, mask
            weight = weight.view(*dim[1:]).numpy()
            weight = map_coordinates(weight, indices, order=1)
            weight = weight.reshape(dim)
            weight = torch.Tensor(weight)

        return (image, mask) if weight is None else (image, mask, weight)


class DoubleCompose(transforms.Compose):

    def __call__(self, image, mask, weight=None):
        if weight is None:
            for t in self.transforms:
                image, mask = t(image, mask)
            return image, mask
        for t in self.transforms:
            image, mask, weight = t(image, mask, weight)
        return image, mask, weight

###############################################################################
# For testing
###############################################################################
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from PIL import Image
    from pathlib import Path

    images = list(Path('data/skinData/trainx/').iterdir())[1:]
    #print(images)
    
    img_path = images[0]
    #print(img_path)


    # the first image data
    img = Image.open(img_path)

    masks = list(Path('data/skinData/trainy').iterdir())[1:]
    # corresponding mask image
    msk_path = masks[0]
    #print(msk_path)
    mask = Image.open(msk_path)

    print("image tensor")
    print(TF.to_tensor(img))

    print("mask tensor")
    print(TF.to_tensor(mask))

    print(img.size)
    print(mask.size)    

    print(TF.to_tensor(img).shape)
    print(TF.to_tensor(mask).shape)

    # mean = 0.495
    # std = 0.173
    # out_size = 388

    image_mask_transform = DoubleCompose([
        DoubleToTensor(),
        DoubleElasticTransform(alpha=250, sigma=10),
        DoubleHorizontalFlip(),
        DoubleVerticalFlip(),
        DoubleCropAndPad(),
    ])

    image_t, mask_t = image_mask_transform(img, mask)
    image_t, mask_t = image_t.numpy()[0], mask_t.numpy()[0]

    # print(image_t.shape)
    # print(mask_t.shape)

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    imgplot = plt.imshow(img)
    ax.set_title('Image')
    ax = fig.add_subplot(2, 2, 2)
    imgplot = plt.imshow(image_t)
    ax.set_title('Transformed Image')
    ax = fig.add_subplot(2, 2, 3)
    imgplot = plt.imshow(mask, cmap='gray')
    ax.set_title('Label')
    ax = fig.add_subplot(2, 2, 4)
    imgplot = plt.imshow(mask_t, cmap='gray')
    ax.set_title('Transformed Label')
    fig.tight_layout()

    plt.show()
    plt.savefig("skin_allaAgmentedData_prob.png")



    '''
    Each Augmentation illustration
    '''
    #plot for every augmentation 
    Horizontal = DoubleCompose([
        DoubleToTensor(),
        DoubleHorizontalFlip(),
    ])
    Elastic = DoubleCompose([
        DoubleToTensor(),
        DoubleElasticTransform(),
    ])
    Vertical = DoubleCompose([
        DoubleToTensor(),
        DoubleVerticalFlip(),
    ])

    CropPad = DoubleCompose([
        DoubleToTensor(),
        DoubleCropAndPad(),
    ])

    image_c, mask_c = CropPad(img, mask)
    image_c, mask_c = image_c.numpy()[0], mask_c.numpy()[0]

    image_h, mask_h = Horizontal(img, mask)
    image_h, mask_h = image_h.numpy()[0], mask_h.numpy()[0]

    image_v, mask_v = Vertical(img, mask)
    image_v, mask_v = image_v.numpy()[0], mask_v.numpy()[0]

    image_e, mask_e = Elastic(img, mask)
    image_e, mask_e = image_e.numpy()[0], mask_e.numpy()[0]


    '''
    Individual plots for each augmentation
    '''
    # random crop and pad
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    imgplot = plt.imshow(img)
    ax.set_title('Image')
    ax = fig.add_subplot(2, 2, 2)
    imgplot = plt.imshow(image_c)
    ax.set_title('Transformed Image')
    ax = fig.add_subplot(2, 2, 3)
    imgplot = plt.imshow(mask, cmap='gray')
    ax.set_title('Label')
    ax = fig.add_subplot(2, 2, 4)
    imgplot = plt.imshow(mask_c, cmap='gray')
    ax.set_title('Transformed Label')
    fig.tight_layout()

    plt.show()
    plt.savefig("skin_CropAndPad.png")

    # horizontal flip
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    imgplot = plt.imshow(img)
    ax.set_title('Image')
    ax = fig.add_subplot(2, 2, 2)
    imgplot = plt.imshow(image_h)
    ax.set_title('Transformed Image')
    ax = fig.add_subplot(2, 2, 3)
    imgplot = plt.imshow(mask, cmap='gray')
    ax.set_title('Label')
    ax = fig.add_subplot(2, 2, 4)
    imgplot = plt.imshow(mask_h, cmap='gray')
    ax.set_title('Transformed Label')
    fig.tight_layout()

    plt.show()
    plt.savefig("skin_HorizontalFlip.png")

    # veritical flip
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    imgplot = plt.imshow(img)
    ax.set_title('Image')
    ax = fig.add_subplot(2, 2, 2)
    imgplot = plt.imshow(image_v)
    ax.set_title('Transformed Image')
    ax = fig.add_subplot(2, 2, 3)
    imgplot = plt.imshow(mask, cmap='gray')
    ax.set_title('Label')
    ax = fig.add_subplot(2, 2, 4)
    imgplot = plt.imshow(mask_v, cmap='gray')
    ax.set_title('Transformed Label')
    fig.tight_layout()

    plt.show()
    plt.savefig("skin_VerticalFlip.png")

    # Elastic Noise
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    imgplot = plt.imshow(img, cmap='gray')
    ax.set_title('Image')
    ax = fig.add_subplot(2, 2, 2)
    imgplot = plt.imshow(image_e, cmap='gray')
    ax.set_title('Transformed Image')
    ax = fig.add_subplot(2, 2, 3)
    imgplot = plt.imshow(mask, cmap='gray')
    ax.set_title('Label')
    ax = fig.add_subplot(2, 2, 4)
    imgplot = plt.imshow(mask_e, cmap='gray')
    ax.set_title('Transformed Label')
    fig.tight_layout()
    plt.show()
    plt.savefig("skin_ElasticNoise.png")


