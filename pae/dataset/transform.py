from math import floor

import torch
from torch import distributions
from torchvision import transforms
from timm.data import rand_augment_transform


def _dynamically_binarize(x):
    return distributions.Bernoulli(probs=x).sample()


def gauss_noise_tensor(img, sigma=0.01):
    return img + sigma * torch.randn_like(img)


class TrainTransform:

    def __init__(self,
                 resize:tuple=(224, 224),
                 resize_mode:str='RandomResizedCrop',
                 pad:int=0,
                 scale:tuple=(0.08, 1.0),
                 ratio:tuple=(0.75, 1.333333),
                 hflip:float=0.5,
                 auto_aug:bool=False,
                 remode:bool=0.2,
                 gaussian_noise=False,
                 interpolation:str='bicubic',
                 mean:tuple=(0.485, 0.456, 0.406),
                 std:tuple=(0.229, 0.224, 0.225),
                 binary_img=False,
                 ):
        interpolation = transforms.functional.InterpolationMode(interpolation)
        augmentation = list()
        corrupt_and_normalize = list()

        if hflip:
            augmentation.append(transforms.RandomHorizontalFlip(hflip))

        if auto_aug:
            augmentation.append(rand_augment_transform('rand-m9-mstd0.5', {}))

        if resize_mode == 'RandomResizedCrop':
            augmentation.append(
                transforms.RandomResizedCrop(resize, scale=scale, ratio=ratio, interpolation=interpolation))
        elif resize_mode == 'ResizeRandomCrop':
            augmentation.extend([transforms.Resize(resize, interpolation=interpolation),
                                 transforms.RandomCrop(resize, padding=pad)])

        augmentation.append(transforms.ToTensor())

        if remode:
            corrupt_and_normalize.append(transforms.RandomErasing(remode))

        if gaussian_noise:
            corrupt_and_normalize.append(gauss_noise_tensor)

        if binary_img:
            corrupt_and_normalize.append(_dynamically_binarize)
        else:
            corrupt_and_normalize.append(transforms.Normalize(mean=mean, std=std))

        self.binary_img = binary_img
        self.augmentation_fn = transforms.Compose(augmentation)
        self.corrupt_and_normalize = transforms.Compose(corrupt_and_normalize)


    def __call__(self, x):
        x = self.augmentation_fn(x)

        if self.binary_img:
            x_in = self.corrupt_and_normalize(x)
            x_out = _dynamically_binarize(x)
        else:
            x_in = self.corrupt_and_normalize(x)
            x_out = (x * 255).long()

        return x_in, x_out


class ValTransform:

    def __init__(self,
                 size:tuple=(224, 224),
                 resize_mode:str='resize_shorter',
                 crop_ptr:float=0.875,
                 interpolation:str='bicubic',
                 mean:tuple=(0.485, 0.456, 0.406),
                 std:tuple=(0.229, 0.224, 0.225),
                 binary_img = False,
                 ):
        interpolation = transforms.functional.InterpolationMode(interpolation)

        if not isinstance(size, (tuple, list)):
            size = (size, size)

        resize = (int(floor(size[0] / crop_ptr)), int(floor(size[1] / crop_ptr)))

        if resize_mode == 'resize_shorter':
            resize = resize[0]

        transform_list = []

        transform_list.extend([
            transforms.Resize(resize, interpolation=interpolation),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])
        self.augmentation_fn = transforms.Compose(transform_list)

        self.binary_img = binary_img
        if binary_img:
            self.normalization = transforms.Compose([_dynamically_binarize])
        else:
            self.normalization = transforms.Compose([transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        x = self.augmentation_fn(x)

        if self.binary_img:
            x_in = self.normalization(x)
            x_out = x_in
        else:
            x_in = self.normalization(x)
            x_out = (x * 255).long()

        return x_in, x_out