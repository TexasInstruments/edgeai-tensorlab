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

import numpy as np
import cv2
from typing import Any, List, Sequence


def crop(img, top: int, left, height, width):
    """Crop the given np array.

    Args:
        img (np.ndarray): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        np.ndarray: Cropped image.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img[top:(top + height), left:(left + width), ...]


def resize(img, size, **kwargs):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        dsize (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.
            For compatibility reasons with ``functional_tensor.resize``, if a tuple or list of length 1 is provided,
            it is interpreted as a single int.
        interpolation (int, optional): Desired interpolation. Default is ``PIL.Image.BILINEAR``.

    Returns:
        PIL Image: Resized image.
    """
    interpolation = kwargs.get('interpolation', None) or cv2.INTER_LINEAR
    pad_color = kwargs.get('pad_color', None) or 0

    if not isinstance(img, np.ndarray):
        raise TypeError('img should be numpy array. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Sequence) and len(size) in (1, 2))):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    resize_with_pad = kwargs.get('resize_with_pad', False)
    if isinstance(size, int) or len(size) == 1:
        if resize_with_pad:
            if isinstance(size, Sequence):
                size = size[0]
            #
            w, h = img.shape[1], img.shape[0]
            if (w >= h and w == size) or (h >= w and h == size):
                ow = w
                oh = h
            elif w > h:
                ow = size
                oh = int(size * h / w)
                img = cv2.resize(img, (ow, oh), interpolation=interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                img = cv2.resize(img, (ow, oh), interpolation=interpolation)
            #
            # pad if necessary
            wpad = (size - ow)
            hpad = (size - oh)
            if  isinstance(resize_with_pad, (list, tuple)):
                if "corner" in resize_with_pad:
                    top, left = 0, 0
                    bottom, right = hpad, wpad
            else :
                top = hpad // 2
                bottom = hpad - top
                left = wpad // 2
                right = wpad - left

            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
            border=(left,top,right,bottom)
            return img, border
        else:
            if isinstance(size, Sequence):
                size = size[0]
            #
            w, h = img.shape[1], img.shape[0]
            if (w <= h and w == size) or (h <= w and h == size):
                pass
            if w < h:
                ow = size
                oh = int(size * h / w)
                img = cv2.resize(img, (ow, oh), interpolation=interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                img = cv2.resize(img, (ow, oh), interpolation=interpolation)
            #
            border = (0,0,0,0)
            return img, border
        #
    else:
        if resize_with_pad:
            w, h = img.shape[1],img.shape[0]
            if(size[0]/h < size[1]/w):
                oh = size[0]
                ow = int(size[0] * w / h)
                img = cv2.resize(img, (ow, oh), interpolation=interpolation)
            else:
                ow = size[1]
                oh = int(size[1] * h / w)
                img = cv2.resize(img, (ow, oh), interpolation=interpolation)
            
            # pad if necessary
            wpad = (size[1] - ow)
            hpad = (size[1] - oh)
            if  isinstance(resize_with_pad, (list, tuple)):
                if "corner" in resize_with_pad:
                    top, left = 0, 0
                    bottom, right = hpad, wpad
            else :
                top = hpad // 2
                bottom = hpad - top
                left = wpad // 2
                right = wpad - left

            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
            border=(left,top,right,bottom)
            return img, border
        else:
            border = (0,0,0,0)
            img = cv2.resize(img, dsize=size[::-1], interpolation=interpolation)
            return img, border
    #


def pad(img, padding, fill=0, padding_mode="constant"):
    left   = padding[0]
    top    = padding[1]
    right  = padding[2]
    bottom = padding[3]

    if padding_mode not in ["constant"]:
        raise ValueError("Padding mode should be constant")

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill)
    return img

