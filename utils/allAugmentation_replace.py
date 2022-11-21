# reference from https://github.com/hayashimasa/UNet-PyTorch
# self-modify the vertical flip and elastic transform

"""Model Trainer
author: Masahiro Hayashi
This script defines custom image transformations that simultaneously transform
both images and segmentation masks.
"""
import torchvision.transforms.functional as TF
# from torchvision.transforms import Compose
from torchvision import transforms
import torch
from torch.utils.tensorboard import SummaryWriter
import random
from random import randint
import numpy as np

import torchvision.transforms as T


class DoubleToTensor:
    """Apply horizontal flips to both image and segmentation mask."""

    def __init__(self, p=1):
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

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, image, mask):
        p = random.random()
        #p = 0.1
        if p < self.p:
            #print("horizontal flip")
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'

class DoubleVerticalFlip:
    """Apply vertical flips to both image and segmentation mask."""

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, image, mask):
        p = random.random()
        #p = 0.1
        if p < self.p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'

class DoubleRandomRotation:
    """
    randomly rotate the image by a degree and apply the same operation on the corresponding mask image
    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, image, mask):
        p = random.random()
        #p = 0.1
        if p < self.p:
            # get the parameters for the same implementation on both images
            params = transforms.RandomRotation.get_params(degrees=(0, 10))

            toImage = T.ToPILImage()
            image= TF.rotate(toImage(image).convert("L"), params)
            mask = TF.rotate(toImage(mask).convert("L"), params)

            return TF.to_tensor(image), TF.to_tensor(mask)

        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'


class DoubleGuassianBlur:
    """
    Add some gaussian on the image and apply the same operation on the corresponding mask image
    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, image, mask):
        p = random.random()
        #p = 0.1
        if p < self.p:
            # get the parameters for the same implementation on both images
            gaussianNoise = transforms.GaussianBlur(3, 0.1)
            image = gaussianNoise(image)
            mask = gaussianNoise(mask)

            return image, mask

        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'


class DoubleCropAndPad:
    """
    Apply croping to both the image and mask
    Padding is also applied to return both the images in the required shape
    """

    def __init__(self, p=0.1, height = 520, width = 696):
        self.p = p
        self.height = height
        self.width = width

    def __call__(self, image, mask):
        p = random.random()
        #p = 0.1
        if p < self.p:
            
            toImage = T.ToPILImage()
            params = transforms.RandomResizedCrop.get_params(image, (0.85, 1.0), (1,1))
            padImage= TF.resized_crop(toImage(image).convert("L"), *params, (self.height, self.width))
            padMask = TF.resized_crop(toImage(mask).convert("L"), *params, (self.height, self.width))

            return TF.to_tensor(padImage), TF.to_tensor(padMask)

        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p})'


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

    images = list(Path('data/imgs/').iterdir())[1:]
    print(images)
    
    img_path = images[0]
    print(img_path)


    # the first image data
    img = Image.open(img_path)

    masks = list(Path('data/masks/').iterdir())[1:]
    # corresponding mask image
    msk_path = masks[0]
    print(msk_path)
    mask = Image.open(msk_path)

    print(img.size)
    print(mask.size)    

    # mean = 0.495
    # std = 0.173
    # out_size = 388

    image_mask_transform = DoubleCompose([
        DoubleToTensor(),
        DoubleElasticTransform(alpha=250, sigma=10),
        DoubleHorizontalFlip(),
        DoubleVerticalFlip(),
        DoubleBrightness(),
        #DoubleGaussianNoise()
    ])

    image_t, mask_t = image_mask_transform(img, mask)
    image_t, mask_t = image_t.numpy()[0], mask_t.numpy()[0]

    print(image_t.shape)
    print(mask_t.shape)

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    imgplot = plt.imshow(img)
    ax.set_title('Image')
    ax = fig.add_subplot(2, 2, 2)
    imgplot = plt.imshow(image_t)
    ax.set_title('Transformed Image')
    ax = fig.add_subplot(2, 2, 3)
    imgplot = plt.imshow(mask)
    ax.set_title('Label')
    ax = fig.add_subplot(2, 2, 4)
    imgplot = plt.imshow(mask_t)
    ax.set_title('Transformed Label')
    fig.tight_layout()

    plt.show()
    plt.savefig("augmentedData_combine.png")

    # # only flip
    # # test for horizontal/vertical flip only
    # image_mask_transform_1 = DoubleCompose([
    #     DoubleToTensor(),
    #     DoubleHorizontalFlip(),
    # ])

    # image_t, mask_t = image_mask_transform_1(img, mask)
    # image_t, mask_t = image_t.numpy()[0], mask_t.numpy()[0]

    # print(image_t.shape)
    # print(mask_t.shape)

    # fig = plt.figure()
    # ax = fig.add_subplot(2, 2, 1)
    # imgplot = plt.imshow(img)
    # ax.set_title('Image')
    # ax = fig.add_subplot(2, 2, 2)
    # imgplot = plt.imshow(image_t)
    # ax.set_title('Transformed Image')
    # ax = fig.add_subplot(2, 2, 3)
    # imgplot = plt.imshow(mask)
    # ax.set_title('Label')
    # ax = fig.add_subplot(2, 2, 4)
    # imgplot = plt.imshow(mask_t)
    # ax.set_title('Transformed Label')
    # fig.tight_layout()

    # plt.show()
    # plt.savefig("augmentedData_flipOnly.png")

    transformsChoice = [
        DoubleElasticTransform(alpha=250, sigma=10),
        DoubleHorizontalFlip(),
        DoubleVerticalFlip(),
        DoubleBrightness(),   
        #DoubleGaussianNoise()
        ]

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
    Brightness = DoubleCompose([
        DoubleToTensor(),
        DoubleBrightness(),
    ])
    # Gaussian = DoubleCompose([
    #     DoubleToTensor(),
    #     DoubleGaussianNoise(),
    # ])

    image_h, mask_h = Horizontal(img, mask)
    image_h, mask_h = image_h.numpy()[0], mask_h.numpy()[0]

    image_v, mask_v = Vertical(img, mask)
    image_v, mask_v = image_v.numpy()[0], mask_v.numpy()[0]

    image_e, mask_e = Elastic(img, mask)
    image_e, mask_e = image_e.numpy()[0], mask_e.numpy()[0]

    # image_g, mask_g = Gaussian(img, mask)
    # image_g, mask_g = image_g.numpy()[0], mask_g.numpy()[0]

    image_b, mask_b = Brightness(img, mask)
    image_b, mask_b = image_b.numpy()[0], mask_b.numpy()[0]

    # print(image_t.shape)
    # print(mask_t.shape)

    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1)
    imgplot = plt.imshow(img)
    ax.set_title('Original Image')

    ax = fig.add_subplot(2, 3, 2)
    imgplot = plt.imshow(image_h)
    ax.set_title('Horizontal T')

    ax = fig.add_subplot(2, 3, 3)
    imgplot = plt.imshow(image_v)
    ax.set_title('Vertical T')

    ax = fig.add_subplot(2, 3, 4)
    imgplot = plt.imshow(mask)
    ax.set_title('Original Mask')

    ax = fig.add_subplot(2, 3, 5)
    imgplot = plt.imshow(mask_h)
    ax.set_title('Horizontal T')

    ax = fig.add_subplot(2, 3, 6)
    imgplot = plt.imshow(mask_v)
    ax.set_title('Vertical T')

    fig.tight_layout()
    plt.show()
    plt.savefig("Augmentation_samplePlots_1.png")


    fig = plt.figure()
    ax = fig.add_subplot(2, 3, 1)
    imgplot = plt.imshow(img)
    ax.set_title('Original Image')

    ax = fig.add_subplot(2, 3, 2)
    imgplot = plt.imshow(image_e)
    ax.set_title('Elastic T')

    # ax = fig.add_subplot(2, 4, 3)
    # imgplot = plt.imshow(image_g)
    # ax.set_title('Gaussian T')

    ax = fig.add_subplot(2, 3, 3)
    imgplot = plt.imshow(image_b)
    ax.set_title('Brightness T')

    ax = fig.add_subplot(2, 3, 4)
    imgplot = plt.imshow(mask)
    ax.set_title('Original Mask')

    ax = fig.add_subplot(2, 3, 5)
    imgplot = plt.imshow(mask_e)
    ax.set_title('Elastic T')

    # ax = fig.add_subplot(2, 4, 7)
    # imgplot = plt.imshow(mask_g)
    # ax.set_title('Gaussian T')

    ax = fig.add_subplot(2, 3, 6)
    imgplot = plt.imshow(mask_b)
    ax.set_title('Brightness T')    

    fig.tight_layout()
    plt.show()
    plt.savefig("Augmentation_samplePlots_2.png")
