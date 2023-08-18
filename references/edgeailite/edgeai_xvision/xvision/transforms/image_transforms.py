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
# Some parts of the code are borrowed from: https://github.com/ansleliu/LightNet
# with the following license:
#
# MIT License
#
# Copyright (c) 2018 Huijun Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#################################################################################

from __future__ import division
import random
import numbers
import math
import cv2
import numpy as np
import PIL

from .image_transform_utils import *


class ConditionalImageTransform(object):
    """
    Just a wrapper to handle None transform
    """
    def __init__(self, t, condition=True):
        super().__init__()
        self.t = t
        self.condition = condition
    
    def forward(self, image, target, condition=True):
        return self.t(image, target) if self.t and self.condition and condition else (image, target)
    

class BypassImages(object):
    def __call__(self, images, targets):
        assert isinstance(images, (list, tuple)), 'Input must a list'
        assert isinstance(targets, (list, tuple)), 'Target must a list'
        return images, targets
    

class CheckImages(object):
    def __call__(self, images, targets):
        assert isinstance(images, (list, tuple)), 'Input must a list'
        assert isinstance(targets, (list, tuple)), 'Target must a list'
        #assert images[0].shape[:2] == targets[0].shape[:2], 'Image and target sizes must match.'

        for img_idx in range(len(images)):
            assert images[img_idx].shape[:2] == images[0].shape[:2], 'Image sizes must match. Either provide same size images or use AlignImages() instead of CheckImages()'
            images[img_idx] = images[img_idx][...,np.newaxis] if (images[img_idx].shape) == 2 else images[img_idx]

        for img_idx in range(len(targets)):
            assert targets[img_idx].shape[:2] == targets[0].shape[:2], 'Target sizes must match. Either provide same size targets or use AlignImages() instead of CheckImages()'
            targets[img_idx] = targets[img_idx][...,np.newaxis] if (targets[img_idx].shape) == 2 else targets[img_idx]

        return images, targets


class AlignImages(object):
    """Resize everything to the first image's size, before transformations begin.
       Also make sure the images are in the desired format."""
    def __init__(self, is_flow=None, interpolation=-1):
        self.is_flow = is_flow
        self.interpolation = interpolation

    def __call__(self, images, targets):
        images = images if isinstance(images, (list,tuple)) else [images]
        images = [np.array(img) if isinstance(img,PIL.Image.Image) else img for img in images]

        targets = targets if isinstance(targets, (list,tuple)) else [targets]
        targets = [np.array(tgt) if isinstance(tgt,PIL.Image.Image) else tgt for tgt in targets]

        img_size = images[0].shape[:2]
        images, targets = Scale(img_size, is_flow=self.is_flow, interpolation=self.interpolation)(images, targets)
        CheckImages()(images, targets)
        return images, targets


class ReverseImageChannels(object):
    """Reverse the channels fo the tensor. eg. RGB to BGR
    """
    def __call__(self, images, targets):
        def func(imgs, img_idx):
            imgs = [ImageTransformUtils.reverse_channels(img_plane) for img_plane in imgs] \
                if isinstance(imgs, list) else ImageTransformUtils.reverse_channels(imgs)
            return imgs

        images = ImageTransformUtils.apply_to_list(func, images)
        # do not apply to targets
        return images, targets


class ConvertToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __call__(self, images, targets):
        def func(imgs, img_idx):
            imgs = [ImageTransformUtils.array_to_tensor(img_plane) for img_plane in imgs] \
                if isinstance(imgs, list) else ImageTransformUtils.array_to_tensor(imgs)
            return imgs

        images = ImageTransformUtils.apply_to_list(func, images)
        targets = ImageTransformUtils.apply_to_list(func, targets)
        return images, targets


class CenterCrop(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size"""
    def __init__(self, size):
        self.size = (int(size), int(size)) if isinstance(size, numbers.Number) else size

    def __call__(self, images, targets):
        def func(img, tgt, img_idx):
            th, tw = self.size
            h1, w1, _ = img.shape
            x1 = int(round((w1 - tw) / 2.))
            y1 = int(round((h1 - th) / 2.))
            img = img[y1: y1 + th, x1: x1 + tw]
            tgt = tgt[y1: y1 + th, x1: x1 + tw]
            return img, tgt

        images = ImageTransformUtils.apply_to_list(func, images)
        targets = ImageTransformUtils.apply_to_list(func, targets)

        return images, targets


class ScaleMinSide(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    After scaling, 'size' will be the size of the smaller edge.
    For example, if height > width, then image will be rescaled to (size * height / width, size)"""
    def __init__(self, size, is_flow=None, interpolation=-1):
        self.size = size
        self.is_flow = is_flow
        self.interpolation = interpolation

    def __call__(self, images, targets):
        def func(img, img_idx, interpolation, is_flow):
            h, w, _ = img.shape
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                ratio = 1.0
                size_out = (h, w)
            else:
                if w < h:
                    ratio = self.size / w
                    size_out = (int(round(ratio * h)), self.size)
                else:
                    ratio = self.size / h
                    size_out = (self.size, int(round(ratio * w)))
            #

            img = ImageTransformUtils.resize_img(img, size_out, interpolation=interpolation, is_flow=is_flow)
            return img

        def func_img(img, img_idx, interpolation=-1):
            is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
            return func(img, img_idx, interpolation=interpolation, is_flow=is_flow_img)

        def func_tgt(img, img_idx):
            is_flow_tgt = (self.is_flow[1][img_idx] if self.is_flow else self.is_flow)
            return func(img, img_idx, interpolation=cv2.INTER_NEAREST, is_flow=is_flow_tgt)

        images = ImageTransformUtils.apply_to_list(func_img, images, interpolation=self.interpolation)
        targets = ImageTransformUtils.apply_to_list(func_tgt, targets)

        return images, targets


class RandomCrop(object):
    """Crops the given images"""
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, images, targets):
        size_h, size_w, _ = images[0].shape
        th, tw = self.size
        x1 = np.random.randint(0, size_w - tw) if (size_w>tw) else 0
        y1 = np.random.randint(0, size_h - th) if (size_h>th) else 0

        def func(img, img_idx):
            return img[y1:y1+th, x1:x1+tw]

        images = ImageTransformUtils.apply_to_list(func, images)
        targets = ImageTransformUtils.apply_to_list(func, targets)

        return images, targets


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given images"""
    def __init__(self, is_flow=None):
        self.is_flow = is_flow

    def __call__(self, images, targets):
        def func(img, img_idx, is_flow):
            img = np.copy(np.fliplr(img))
            if is_flow:
                img = ImageTransformUtils.scale_flow(img, (-1), 1)
            return img

        def func_img(img, img_idx):
            is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
            img = func(img, img_idx, is_flow_img)
            return img

        def func_tgt(img, img_idx):
            is_flow_tgt = (self.is_flow[1][img_idx] if self.is_flow else self.is_flow)
            img = func(img, img_idx, is_flow_tgt)
            return img

        if np.random.random() < 0.5:
            images = ImageTransformUtils.apply_to_list(func_img, images)
            targets = ImageTransformUtils.apply_to_list(func_tgt, targets)

        return images, targets


class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=None):
        self.is_flow = is_flow

    def __call__(self, images, targets):
        def func(img, img_idx, is_flow):
            img = np.copy(np.flipud(img))
            if is_flow:
                img = ImageTransformUtils.scale_flow(img, 1, (-1))
            return img

        def func_img(img, img_idx):
            is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
            img = func(img, img_idx, is_flow_img)
            return img

        def func_tgt(img, img_idx):
            is_flow_tgt = (self.is_flow[1][img_idx] if self.is_flow else self.is_flow)
            img = func(img, img_idx, is_flow_tgt)
            return img

        if np.random.random() < 0.5:
            images = ImageTransformUtils.apply_to_list(func_img, images)
            targets = ImageTransformUtils.apply_to_list(func_tgt, targets)

        return images, targets


class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """
    def __init__(self, angle, diff_angle=0, is_flow=None):
        self.angle = angle
        self.diff_angle = diff_angle #if diff_angle else min(angle/2,10)
        self.is_flow = is_flow

    def __call__(self, images, targets):
        applied_angle = random.uniform(-self.angle,self.angle)
        is_input_image_pair = (len(images) == 2) and ((self.is_flow == None) or (not np.any(self.is_flow[0])))
        diff = random.uniform(-self.diff_angle,self.diff_angle) if is_input_image_pair else 0
        angles = [applied_angle - diff/2, applied_angle + diff/2] if is_input_image_pair else [applied_angle for img in images]

        def func(img, img_idx, angle, interpolation, is_flow):
            h, w = img.shape[:2]
            angle_rad = (angle * np.pi / 180)

            if is_flow:
                img = img.astype(np.float32)
                diff_rad = (diff * np.pi / 180)
                def rotate_flow(i, j, k):
                    return -k * (j - w / 2) * diff_rad + (1 - k) * (i - h / 2) * diff_rad
                #
                rotate_flow_map = np.fromfunction(rotate_flow, img.shape)
                img += rotate_flow_map

            img = ImageTransformUtils.rotate_img(img, angle, interpolation)

            # flow vectors must be rotated too! careful about Y flow which is upside down
            if is_flow:
                img = np.copy(img)
                img[:,:,0] = np.cos(angle_rad)*img[:,:,0] + np.sin(angle_rad)*img[:,:,1]
                img[:,:,1] = -np.sin(angle_rad)*img[:,:,0] + np.cos(angle_rad)*img[:,:,1]

            return img

        def func_img(img, img_idx):
            is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
            interpolation = (cv2.INTER_NEAREST if is_flow_img else cv2.INTER_LINEAR)
            img = func(img, img_idx, angles[img_idx], interpolation, is_flow_img)
            return img

        def func_tgt(img, img_idx):
            is_flow_tgt = (self.is_flow[1][img_idx] if self.is_flow else self.is_flow)
            interpolation = (cv2.INTER_NEAREST)
            img = func(img, img_idx, applied_angle, interpolation, is_flow_tgt)
            return img

        if np.random.random() < 0.5:
            images = ImageTransformUtils.apply_to_list(func_img, images)
            targets = ImageTransformUtils.apply_to_list(func_tgt, targets)

        return images, targets


class RandomColorWarp(object):
    def __init__(self, mean_range=0, std_range=0, is_flow=None):
        self.mean_range = mean_range
        self.std_range = std_range
        self.is_flow = is_flow

    def __call__(self, images, target):
        if np.random.random() < 0.5:
            if self.std_range != 0:
                random_std = np.random.uniform(-self.std_range, self.std_range, 3)
                for img_idx in range(len(images)):
                    is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
                    if not is_flow_img:
                        images[img_idx] *= (1 + random_std)

            if self.mean_range != 0:
                random_mean = np.random.uniform(-self.mean_range, self.mean_range, 3)
                for img_idx in range(len(images)):
                    is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
                    if not is_flow_img:
                        images[img_idx] += random_mean

            for img_idx in range(len(images)):
                is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
                if not is_flow_img:
                    random_order = np.random.permutation(3)
                    images[img_idx] = images[img_idx][:,:,random_order]

        return images, target


class RandomColor2Gray(object):
    def __init__(self, mean_range=0, std_range=0, is_flow=None, random_threshold=0.25):
        self.mean_range = mean_range
        self.std_range = std_range
        self.is_flow = is_flow
        self.random_threshold = random_threshold

    def __call__(self, images, target):
        if np.random.random() < self.random_threshold:
            for img_idx in range(len(images)):
                is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
                if not is_flow_img:
                    images[img_idx] = cv2.cvtColor(images[img_idx], cv2.COLOR_RGB2GRAY)
                    images[img_idx] = cv2.cvtColor(images[img_idx], cv2.COLOR_GRAY2RGB)
        return images, target


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""
    def __init__(self, img_resize, scale_range=(1.0,2.0), is_flow=None, center_crop=False, resize_in_yv12=False,
                 interpolation=-1):
        self.img_resize = img_resize
        self.scale_range = scale_range
        self.is_flow = is_flow
        self.center_crop = center_crop
        self.resize_in_yv12 = resize_in_yv12
        self.interpolation = interpolation

    @staticmethod
    def get_params(img, img_resize, scale_range, center_crop, resize_in_yv12 = False):
        in_h, in_w = img.shape[:2]
        out_h, out_w = img_resize
        if resize_in_yv12:
            #to make U,V as multiple of 4 shape to properly represent in YV12 format
            round_or_align4 = lambda x: ((int(x)//4)*4)
        else:
            round_or_align4 = lambda x: round(x)
        # this random scaling is w.r.t. the output size
        if (np.random.random() < 0.5):
            resize_h = int(round_or_align4(np.random.uniform(scale_range[0], scale_range[1]) * out_h))
            resize_w = int(round_or_align4(np.random.uniform(scale_range[0], scale_range[1]) * out_w))
        else:
            resize_h, resize_w = out_h, out_w

        # crop params w.r.t the scaled size
        out_r = (resize_h - out_h)//2 if center_crop else np.random.randint(resize_h - out_h + 1)
        out_c = (resize_w - out_w)//2 if center_crop else np.random.randint(resize_w - out_w + 1)
        return out_r, out_c, out_h, out_w, resize_h, resize_w

    def __call__(self, images, targets):
        out_r, out_c, out_h, out_w, resize_h, resize_w = self.get_params(images[0], self.img_resize, self.scale_range,
                                self.center_crop, resize_in_yv12 = self.resize_in_yv12)

        def func_img(img, img_idx, interpolation=-1):
            is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
            img = ImageTransformUtils.resize_and_crop(img, out_r, out_c, out_h, out_w, (resize_h, resize_w),
                                interpolation=interpolation, is_flow=is_flow_img, resize_in_yv12=self.resize_in_yv12)
            return img

        def func_tgt(img, img_idx):
            is_flow_tgt = (self.is_flow[1][img_idx] if self.is_flow else self.is_flow)
            img = ImageTransformUtils.resize_and_crop(img, out_r, out_c, out_h, out_w, (resize_h, resize_w),
                                interpolation=cv2.INTER_NEAREST, is_flow=is_flow_tgt)
            return img

        images = ImageTransformUtils.apply_to_list(func_img, images, interpolation=self.interpolation)
        targets = ImageTransformUtils.apply_to_list(func_tgt, targets)
        return images, targets


class RandomCropScale(object):
    """Crop the Image to random size and scale to the given resolution"""
    def __init__(self, size, crop_range=(0.08, 1.0), is_flow=None, center_crop=False, resize_in_yv12=False,
                 interpolation=-1):
        self.size = size if (type(size) in (list,tuple)) else (size, size)
        self.crop_range = crop_range
        self.is_flow = is_flow
        self.center_crop = center_crop
        self.resize_in_yv12 = resize_in_yv12
        self.interpolation = interpolation

    @staticmethod
    def get_params(img, crop_range, center_crop):
        h_orig = img.shape[0]; w_orig = img.shape[1]
        r_wh = (w_orig/h_orig)
        ratio =  (r_wh/2.0, 2.0*r_wh)
        for attempt in range(10):
            area = h_orig * w_orig
            target_area = random.uniform(*crop_range) * area
            aspect_ratio = random.uniform(*ratio)
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if (h <= h_orig) and (w <= w_orig):
                i = (h_orig - h)//2 if center_crop else random.randint(0, h_orig - h)
                j = (w_orig - w)//2 if center_crop else random.randint(0, w_orig - w)
                return i, j, h, w

        # Fallback: entire image
        return 0, 0, h_orig, w_orig

    def __call__(self, images, targets):
        out_r, out_c, out_h, out_w = self.get_params(images[0], self.crop_range, self.center_crop)

        def func_img(img, img_idx, interpolation=-1):
            is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
            img = ImageTransformUtils.crop_and_resize(img, out_r, out_c, out_h, out_w, self.size,
                        interpolation=interpolation, is_flow=is_flow_img, resize_in_yv12=self.resize_in_yv12)
            return img

        def func_tgt(img, img_idx):
            is_flow_tgt = (self.is_flow[1][img_idx] if self.is_flow else self.is_flow)
            img = ImageTransformUtils.crop_and_resize(img, out_r, out_c, out_h, out_w, self.size,
                        interpolation=cv2.INTER_NEAREST, is_flow=is_flow_tgt)
            return img

        images = ImageTransformUtils.apply_to_list(func_img, images, interpolation=self.interpolation)
        targets = ImageTransformUtils.apply_to_list(func_tgt, targets)

        return images, targets


class Scale(object):
    def __init__(self, img_size, target_size=None, is_flow=None, interpolation=-1):
        self.img_size = img_size
        self.target_size = target_size if target_size else img_size
        self.is_flow = is_flow
        self.interpolation = interpolation

    def __call__(self, images, targets):
        if self.img_size is None:
            return images, targets

        def func_img(img, img_idx, interpolation=-1):
            is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
            img = ImageTransformUtils.resize_img(img, self.img_size, interpolation=interpolation, is_flow=is_flow_img)
            return img

        def func_tgt(img, img_idx):
            is_flow_tgt = (self.is_flow[1][img_idx] if self.is_flow else self.is_flow)
            img = ImageTransformUtils.resize_img(img, self.target_size, interpolation=cv2.INTER_NEAREST, is_flow=is_flow_tgt)
            return img

        images = ImageTransformUtils.apply_to_list(func_img, images, interpolation=self.interpolation)
        targets = ImageTransformUtils.apply_to_list(func_tgt, targets)

        return images, targets


class CropRect(object):
    def __init__(self, crop_rect):
        self.crop_rect = crop_rect

    def __call__(self, images, targets):
        if self.crop_rect is None:
            return images, targets

        def func(img, tgt, img_idx):
            img_size = img.shape
            crop_rect = self.crop_rect
            max_val = max(crop_rect)
            t, l, h, w = crop_rect
            if max_val <= 1:  # convert float into integer
                t = int(t * img_size[0] + 0.5)  # top
                l = int(l * img_size[1] + 0.5)  # left
                h = int(h * img_size[0] + 0.5)  # height
                w = int(w * img_size[1] + 0.5)  # width
            else:
                t = int(t)  # top
                l = int(l)  # left
                h = int(h)  # height
                w = int(w)  # width

            img = img[t:(t+h), l:(l+w)]
            tgt = tgt[t:(t+h), l:(l+w)]

            return img, tgt

        images, targets = ImageTransformUtils.apply_to_lists(func, images, targets)

        return images, targets


class MaskTarget(object):
    def __init__(self, mask_rect, mask_val):
        self.mask_rect = mask_rect
        self.mask_val = mask_val

    def __call__(self, images, targets):
        if self.mask_rect is None:
            return images, targets

        def func(img, tgt, img_idx):
            img_size = img.shape
            crop_rect = self.mask_rect
            max_val = max(crop_rect)
            t, l, h, w = crop_rect
            if max_val <= 1:  # convert float into integer
                t = int(t * img_size[0] + 0.5)  # top
                l = int(l * img_size[1] + 0.5)  # left
                h = int(h * img_size[0] + 0.5)  # height
                w = int(w * img_size[1] + 0.5)  # width
            else:
                t = int(t)  # top
                l = int(l)  # left
                h = int(h)  # height
                w = int(w)  # width

            tgt[t:(t+h), l:(l+w)] = self.mask_val
            return img, tgt

        images = ImageTransformUtils.apply_to_list(func, images)
        targets = ImageTransformUtils.apply_to_list(func, targets)

        return images, targets


class NormalizeMeanStd(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, target):
        if isinstance(images, (list,tuple)):
            images = [(img-self.mean)/self.std for img in images]
        else:
            images = (images-self.mean)/self.std

        return images, target


class NormalizeMeanScale(object):
    def __init__(self, mean, scale):
        self.mean = mean
        self.scale = scale


    def __call__(self, images, target):
        def func(imgs, img_idx):
            if isinstance(imgs, (list,tuple)):
                imgs = [(img-self.mean)*self.scale for img in imgs]
            else:
                imgs = (imgs-self.mean)*self.scale
            #
            return imgs
        #
        images = ImageTransformUtils.apply_to_list(func, images) \
            if isinstance(images, (list,tuple)) else func(images)
        return images, target
