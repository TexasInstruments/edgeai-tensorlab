# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

################################################################################

# Also includes parts from: https://github.com/pytorch/vision
# License: https://github.com/pytorch/vision/blob/master/LICENSE
#
# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numbers
from collections.abc import Sequence
import numpy as np
import PIL
import cv2

from PIL import Image
from . import functional as F

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class ImageRead(object):
    def __init__(self, backend='pil'):
        assert backend in ('pil', 'cv2'), f'backend must be one of pil or cv2. got {backend}'
        self.backend = backend

    def __call__(self, path, info_dict):
        if isinstance(path, str):
            img_data = None
            if self.backend == 'pil':
                img_data = PIL.Image.open(path)
                img_data = img_data.convert('RGB')
                info_dict['data_shape'] = img_data.size[1], img_data.size[0], len(img_data.getbands())
            elif self.backend == 'cv2':
                img_data = cv2.imread(path)
                if img_data.shape[-1] == 1:
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                elif img_data.shape[-1] == 4:
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
                #
                # always return in RGB format
                img_data = img_data[:,:,::-1]
                info_dict['data_shape'] = img_data.shape
            #
            info_dict['data'] = img_data
            info_dict['data_path'] = path
        elif isinstance(path, np.ndarray):
            img_data = path
            info_dict['data_shape'] = img_data.shape
            info_dict['data'] = img_data
            info_dict['data_path'] = './'
        else:
            assert False, 'invalid input'
        #
        return img_data, info_dict

    def __repr__(self):
        return self.__class__.__name__ + f'(backend={self.backend})'


class ImageNorm(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, data_layout='NCHW', inplace=False):
        self.mean = mean
        self.std = std
        self.data_layout = data_layout
        self.inplace = inplace

    def __call__(self, tensor, info_dict):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if isinstance(tensor, (list,tuple)):
            tensor = [F.normalize(t, self.mean, self.std, self.data_layout, self.inplace) for t in tensor]
        else:
            tensor = F.normalize(tensor, self.mean, self.std, self.data_layout, self.inplace)
        #
        return tensor, info_dict

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ImageNormMeanScale(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and scale: ``(scale[1],..,scale[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``output[channel] = (input[channel] - mean[channel]) * scale[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        scale (sequence): Sequence of factors for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, scale, data_layout='NCHW', inplace=False):
        self.mean = mean
        self.scale = scale
        self.data_layout = data_layout
        self.inplace = inplace

    def __call__(self, tensor, info_dict):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if isinstance(tensor, (list,tuple)):
            tensor = [F.normalize_mean_scale(t, self.mean, self.scale, self.data_layout, self.inplace) for t in tensor]
        else:
            tensor = F.normalize_mean_scale(tensor, self.mean, self.scale, self.data_layout, self.inplace)
        #
        return tensor, info_dict

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, scale={1})'.format(self.mean, self.scale)


class ImageResize():
    """Resize the input image to the given size.
    The image can be a PIL Image, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).
            In torchscript mode padding as single int is not supported, use a tuple or
            list of length 1: ``[size, ]``.
        interpolation (int, optional): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
    """

    def __init__(self, size, *args, **kwargs):
        super().__init__()
        self.size = size
        self.args = args
        self.kwargs = kwargs

    def __call__(self, img, info_dict):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        img, border = F.resize(img, self.size, *self.args, **self.kwargs)
        if isinstance(img, np.ndarray):
            info_dict['resize_shape'] = img.shape
        else:
            info_dict['resize_shape'] = img.size[1], img.size[0], len(img.getbands())
        #
        info_dict['resize_border'] = border
        return img, info_dict

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'({self.size}'
        for arg in self.args:
            repr_str += f', {arg}'
        #
        for k, v in self.kwargs.items():
            repr_str += f', {k}={v}'
        #
        repr_str += ')'
        return repr_str


class ImageCenterCrop():
    """Crops the given image at the center.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, size=None):
        super().__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        elif isinstance(size, Sequence) and len(size) == 1:
            self.size = (size[0], size[0])
        elif size is None:
            self.size = size
        else:
            if len(size) != 2:
                raise ValueError("Please provide only two dimensions (h, w) for size.")
            self.size = size
        #

    def __call__(self, img, info_dict):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return F.center_crop(img, self.size) if self.size is not None else img, info_dict

    def __repr__(self):
        if self.size is not None:
            return self.__class__.__name__ + '(size={0})'.format(self.size)
        else:
            return self.__class__.__name__ + '()'


class ImageToNPTensor(object):
    """Convert a ``Image`` to a tensor of the same type.

    Converts a PIL Image or numpy array (H x W x C) to a numpy Tensor of shape (C x H x W).
    """

    def __init__(self, data_layout='NCHW', reverse_channels=False):
        self.data_layout = data_layout
        self.reverse_channels = reverse_channels

    def __call__(self, pic, info_dict):
        """
        Args:
            pic (PIL Image): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_numpy_tensor(pic, self.data_layout, self.reverse_channels), info_dict

    def __repr__(self):
        return self.__class__.__name__ + f'({self.data_layout}, {self.reverse_channels})'


class ImageToNPTensor4D(object):
    """Convert a ``Image`` to a tensor of the same type.

    Converts a PIL Image or numpy array (H x W x C) to a numpy Tensor of shape (C x H x W).
    """

    def __init__(self, data_layout='NCHW', reverse_channels=False):
        self.data_layout = data_layout
        self.reverse_channels = reverse_channels

    def __call__(self, pic, info_dict):
        """
        Args:
            pic (PIL Image): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_numpy_tensor_4d(pic, self.data_layout, self.reverse_channels), info_dict

    def __repr__(self):
        return self.__class__.__name__ + f'({self.data_layout}, {self.reverse_channels})'


class NPTensor4DChanReverse(object):
    """Convert a ``Image`` to a tensor of the same type.

    Converts a PIL Image or numpy array (H x W x C) to a numpy Tensor of shape (C x H x W).
    """

    def __init__(self, data_layout='NCHW'):
        assert data_layout in ('NCHW', 'NHWC'), f'invalid data_layout {data_layout}'
        self.data_layout = data_layout

    def __call__(self, pic, info_dict):
        """
        Args:
            pic (np.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted tensor.
        """
        if self.data_layout == 'NCHW':
            return pic[:,::-1,...], info_dict
        else:
            return pic[...,::-1], info_dict

    def __repr__(self):
        return self.__class__.__name__ + f'({self.data_layout})'

class ImageFlipAdd:
    def __init__(self, flip_axis = 3):
        self.flip_axis = flip_axis

    def __call__(self, img, info_dict):
        info_dict['flip_img'] = np.flip(img, axis=[self.flip_axis])
        return img, info_dict