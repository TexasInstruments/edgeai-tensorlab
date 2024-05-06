import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    list_size = isinstance(size, (list,tuple))
    min_size = min(img.size)
    if list_size:
        if img.size[0]<size[1] or img.size[1]<size[0]:
            ow, oh = img.size
            padh = size[0] - oh if oh < size[0] else 0
            padw = size[1] - ow if ow < size[1] else 0
            img = F.pad(img, (0, 0, padw, padh), fill=fill)
        #
    elif min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        list_size = isinstance(self.min_size, (list,tuple))
        if list_size:
            size = (random.randint(self.min_size[0], self.max_size[0]), random.randint(self.min_size[1], self.max_size[1]))
        else:
            size = random.randint(self.min_size, self.max_size)

        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        list_size = isinstance(self.size, (list,tuple))
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, self.size if list_size else (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class NormalizeMeanScale(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        scale (sequence): Sequence of scales for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, scale, inplace=False):
        super().__init__()
        self.mean = mean
        self.scale = scale
        self.inplace = inplace

    def forward(self, tensor, target):
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        tensor = tensor if self.inplace else tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.scale):
            t.sub_(m).mul_(s)

        return tensor, target

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, scale={1})'.format(self.mean, self.scale)

