#################################################################################
# Copyright (c) 2018-2021, Texas Instruments Incorporated - http://www.ti.com
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
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
#
#################################################################################


import math
import random
from collections.abc import Sequence
from typing import Tuple, List, Optional

import torch
from PIL import Image
from torch import Tensor

import numpy as np


class ConditionalTransform(torch.nn.Module):
    """
    Just a wrapper to handle None transform
    """
    def __init__(self, t, condition=True):
        super().__init__()
        self.t = t
        self.condition = condition
    
    def forward(self, img, condition=True):
        return self.t(img) if self.t and self.condition and condition else img


class MultiColor(torch.nn.Module):
    """Convert image to multiple color spaces.

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        list of PIL Images

    """

    def __init__(self, colors=('rgb','yuv','hsv','lab'), concatenate=True):
        super().__init__()
        self.colors = colors
        self.concatenate = concatenate

    def forward(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            list of PIL Images
        """

        images = []

        if ('rgb' in self.colors):
            rgb = img
            images.append(rgb)

        img_arr = np.asarray(img)

        if ('yuv' in self.colors) or ('ycbcr' in self.colors):
            #yuv = img.convert('YCbCr')
            yuv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2YUV)
            images.append(yuv)

        if ('hsv' in self.colors):
            #hsv = img.convert('HSV')
            hsv = cv2.cvtColor(img_arr, cv2.COLOR_RGB2HSV)
            images.append(hsv)

        if ('lab' in self.colors):
            #lab = img.convert('LAB') # not working
            #lab = skimage.color.rgb2lab(img).astype(np.float32) # slow
            lab = cv2.cvtColor(img_arr, cv2.COLOR_RGB2Lab)
            images.append(lab)

        if self.concatenate:
            images = np.concatenate(images,axis=2)

        return images

    def __repr__(self):
        return self.__class__.__name__ + '(colors={0})'.format(self.colors)



class Bypass(torch.nn.Module):
    """Bypass.

    Args:
        None
    """

    def __init__(self):
        super().__init__()

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image

        Returns:
            PIL Image: returns the same image
        """
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ReverseChannels(torch.nn.Module):
    """Reverse the channels fo the tensor. eg. RGB to BGR
    """
    def forward(self, pic):
        """
        Args:
            image (PIL.Image)

        Returns:
            image: Converted image.
        """
        return Image.fromarray(np.array(pic)[:,:,::-1])

    def __repr__(self):
        return self.__class__.__name__ + '()'



class ToFloat(torch.nn.Module):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to float tensor.

    Note: Once the tensor is converted to float, the transform ToTensor will not divide by 255. It only does that with ByteTensor
    """

    def forward(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, (list,tuple)):
            tensor = [np.array(p, dtype=np.float32) for p in pic]
        else:
            tensor = np.array(pic, dtype=np.float32)

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NormalizeMeanScale(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, scale):
        super().__init__()
        self.mean = mean
        self.scale = scale

    def normalize_mean_scale(self, tensor, mean, scale):
        if not isinstance(mean, (list, tuple)):
            mean = [mean]*tensor.size(0)

        if not isinstance(scale, (list, tuple)):
            scale = [scale]*tensor.size(0)

        assert (tensor.size(0) == len(mean)) and (tensor.size(0) == len(scale)), \
            "mean and scale must be boradcastable to the input shape: {} vs {} and {}".format(tensor.shape, mean, scale)

        # This is faster than using broadcasting, don't change without benchmarking
        for t, m, s in zip(tensor, mean, scale):
            t.sub_(m).mul_(s)
        return tensor

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if isinstance(tensor, (list,tuple)):
            tensor = [self.normalize_mean_scale(t, self.mean, self.scale) for t in tensor]
        else:
            tensor = self.normalize_mean_scale(tensor, self.mean, self.scale)

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, scale={1})'.format(self.mean, self.scale)



