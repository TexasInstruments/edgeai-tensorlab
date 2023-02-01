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
# Some parts of the code are borrowed from: https://github.com/pytorch/vision
# with the following license:
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
#
#################################################################################


import numpy as np
import cv2
import torch
import types
import PIL

class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, t):
        self.co_transforms = t

    def extend(self, t):
        self.co_transforms.extend(t)

    def insert(self, index, t):
        self.co_transforms.insert(index, t)

    def write_img(self, img=[], ch_num=-1, name='', en=False):
        if en == False:
            return
        #name = './data/checkpoints/tiad_interest_pt_descriptor/debug/{:02d}.jpg'.format(aug_idx)
        scale_range = 255.0 / np.max(img)
        img = np.clip(img * scale_range, 0.0, 255.0)
        img = np.asarray(img, 'uint8')

        non_zero_el = cv2.countNonZero(img)

        print("non zero element: {}".format(non_zero_el))
        cv2.imwrite('{}_nz{}.jpg'.format(name, non_zero_el), img)

    def __call__(self, input, target):
        if self.co_transforms:
            for aug_idx, t in enumerate(self.co_transforms):
                if t:
                    input,target = t(input,target)

        return input,target



class Bypass(object):
    def __init__(self):
        pass

    def __call__(self, images, targets):
        return images,targets


class Lambda(object):
    """Applies a lambda as a transform"""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, input,target):
        return self.lambd(input,target)


class ImageTransformUtils(object):
    @staticmethod
    def apply_to_list(func, inputs, *args, **kwargs):
        for img_idx in range(len(inputs)):
            inputs[img_idx] = func(inputs[img_idx], img_idx, *args, **kwargs)

        return inputs

    @staticmethod
    def apply_to_lists(func, images, targets, *args, **kwargs):
        for img_idx in range(len(images)):
            images[img_idx], targets[img_idx] = func(images[img_idx], targets[img_idx], img_idx, *args, **kwargs)

        return images, targets

    @staticmethod
    def crop(img, r, c, h, w):
        img = img[r:(r+h), c:(c+w),...] if (len(img.shape)>2) else img[r:(r+h), c:(c+w)]
        return img

    @staticmethod
    def resize_fast(img, output_size_rc, interpolation=-1):
        in_h, in_w = img.shape[:2]
        out_h, out_w = output_size_rc
        if interpolation<0:
            interpolation = cv2.INTER_AREA if ((out_h<in_h) or (out_w<in_w)) else cv2.INTER_LINEAR

        img = cv2.resize(img, (out_w,out_h), interpolation=interpolation) #opencv expects size in (w,h) format
        img = img[...,np.newaxis] if len(img.shape) < 3 else img
        return img

    @staticmethod
    def resize_img(img, size, interpolation=-1, is_flow=False):
        #if (len(img.shape) == 3) and (img.shape[2] == 1 or img.shape[2] == 3):
        #    return __class__.resize_fast(img, size, interpolation)

        in_h, in_w = img.shape[:2]
        out_h, out_w = size
        if interpolation is None or interpolation < 0:
            interpolation = cv2.INTER_AREA if ((out_h<in_h) or (out_w<in_w)) else cv2.INTER_LINEAR

        # opencv handles planar, 1 or 3 channel images
        img = img[...,np.newaxis] if len(img.shape) < 3 else img
        num_chans = img.shape[2]
        img = np.concatenate([img]+[img[...,0:1]]*(3-num_chans), axis=2) if num_chans<3 else img
        img = cv2.resize(img, (out_w, out_h), interpolation=interpolation)
        img = img[...,:num_chans]

        if is_flow:
            ratio_h = out_h / in_h
            ratio_w = out_w / in_w
            img = ImageTransformUtils.scale_flow(img, ratio_w, ratio_h)

        return img

    @staticmethod
    def resize_and_crop(img, r, c, h, w, size, interpolation=-1, is_flow=False, resize_in_yv12=False):
        if resize_in_yv12:
            yv12 = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_YV12)
            yv12 = ImageTransformUtils.resize_img_yv12(yv12, size, interpolation, is_flow)
            img = cv2.cvtColor(yv12, cv2.COLOR_YUV2RGB_YV12)
        else:
            img = ImageTransformUtils.resize_img(img, size, interpolation, is_flow)
        #
        img = ImageTransformUtils.crop(img, r, c, h, w)
        return img

    @staticmethod
    def crop_and_resize(img, r, c, h, w, size, interpolation=-1, is_flow=False, resize_in_yv12=False):
        img = ImageTransformUtils.crop(img, r, c, h, w)
        if resize_in_yv12:
            yv12 = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_YV12)
            yv12 = ImageTransformUtils.resize_img_yv12(yv12, size, interpolation, is_flow)
            img = cv2.cvtColor(yv12, cv2.COLOR_YUV2RGB_YV12)
        else:
            img = ImageTransformUtils.resize_img(img, size, interpolation, is_flow)
        #
        return img

    @staticmethod
    def rotate_img(img, angle, interpolation=-1):
        h, w = img.shape[:2]
        rmat2x3 = cv2.getRotationMatrix2D(center=(w//2,h//2), angle=angle, scale=1.0)
        interpolation = cv2.INTER_NEAREST if interpolation < 0 else interpolation
        img = cv2.warpAffine(img, rmat2x3, (w,h), flags=interpolation)
        return img

    @staticmethod
    def reverse_channels(img):
        if isinstance(img, np.ndarray):
            return img[:,:,::-1]
        elif isinstance(img, PIL.Image):
            return PIL.Image.fromarray(np.array(img)[:,:,::-1])
        else:
            assert False, 'unrecognized image type'

    @staticmethod
    def array_to_tensor(array):
        """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
        assert(isinstance(array, np.ndarray))
        # put it from HWC to CHW format
        array = np.transpose(array, (2, 0, 1))
        if len(array.shape) < 3:
            array = array[np.newaxis, ...]
        #
        tensor = torch.from_numpy(array)
        return tensor.float()

    @staticmethod
    def scale_flow(flow, ratio_x, ratio_y):
        flow = flow.astype(np.float32)
        flow[...,0] *= ratio_x
        flow[...,1] *= ratio_y
        return flow

    @staticmethod
    def scale_flows(inputs, ratio_x, ratio_y, is_flow):
        for img_idx in range(len(inputs)):
            if is_flow and is_flow[img_idx]:
                inputs[img_idx] = inputs[img_idx].astype(np.float32)
                inputs[img_idx][...,0] *= ratio_x
                inputs[img_idx][...,1] *= ratio_y
        #
        return inputs


    #############################################################
    # functions for nv12

    @staticmethod
    def resize_img_yv12(img, size, interpolation=-1, is_flow=False):
        #if (len(img.shape) == 3) and (img.shape[2] == 1 or img.shape[2] == 3):
        #    return __class__.resize_fast(img, size, interpolation)
        debug_print = False
        in_w = img.shape[1]
        in_h = (img.shape[0] * 2) // 3
        y_h = in_h
        uv_h = in_h // 4
        u_w = in_w // 2

        Y = img[0:in_h, 0:in_w]
        V = img[y_h:y_h + uv_h, 0:in_w]
        #print(V[0:2,0:8])
        #print(V[0:2, u_w:u_w+8])
        V = V.reshape(V.shape[0]*2, -1)
        #print(V[0:2, 0:8])
        #print(V[0:2, u_w:u_w + 8])
        U = img[y_h + uv_h:y_h + 2 * uv_h, 0:in_w]
        U = U.reshape(U.shape[0] * 2, -1)

        out_h, out_w = size
        if interpolation < 0:
            interpolation = cv2.INTER_AREA if ((out_h < in_h) or (out_w < in_w)) else cv2.INTER_LINEAR

        Y = cv2.resize(Y, (out_w, out_h), interpolation=interpolation)
        U = cv2.resize(U, (out_w//2, out_h//2), interpolation=interpolation)
        V = cv2.resize(V, (out_w//2, out_h//2), interpolation=interpolation)

        img = np.zeros((out_h*3//2, out_w), dtype='uint8')
        op_uv_h = out_h // 4

        img[0:out_h, 0:out_w] = Y[:, :]
        #print(V[0:2,0:8])
        V = V.reshape(V.shape[0] // 2, -1)
        #print(V[0:1,0:8])
        #print(V[0:1, op_u_w:op_u_w+8])
        img[out_h:out_h + op_uv_h, 0:out_w] = V
        U = U.reshape(U.shape[0] // 2, -1)
        img[out_h + op_uv_h:out_h + 2 * op_uv_h, 0:out_w] = U

        if debug_print:
            h = img.shape[0] * 2 // 3
            w = img.shape[1]
            print("-" * 32, "Resize in YV12")
            print("Y")
            print(img[0:5, 0:5])

            print("V Odd Lines")
            print(img[h:h + 5, 0:5])

            print("V Even Lines")
            print(img[h:h + 5, w // 2:w // 2 + 5])

            print("U Odd Lines")
            print(img[h + h // 4:h + h // 4 + 5, 0:5])

            print("U Even Lines")
            print(img[h + h // 4:h + h // 4 + 5, w // 2:w // 2 + 5])

            print("-" * 32)

        if is_flow:
            ratio_h = out_h / in_h
            ratio_w = out_w / in_w
            img = ImageTransformUtils.scale_flow(img, ratio_w, ratio_h)

        return img


    # @staticmethod
    # def resize_and_crop_yv12(img, r, c, h, w, size, interpolation=-1, is_flow=False):
    #     yv12 = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_YV12)
    #     yv12 = ImageTransformUtils.resize_img_yv12(yv12, size, interpolation, is_flow)
    #     img = cv2.cvtColor(yv12, cv2.COLOR_YUV2RGb_YV12)
    #     img = ImageTransformUtils.crop(img, r, c, h, w)
    #     return img
    #
    # @staticmethod
    # def crop_and_resize_yv12(img, r, c, h, w, size, interpolation=-1, is_flow=False):
    #     img = ImageTransformUtils.crop(img, r, c, h, w)
    #     yv12 = cv2.cvtColor(img, cv2.COLOR_RGB2YUV_YV12)
    #     yv12 = ImageTransformUtils.resize_img_yv12(img, size, interpolation, is_flow)
    #     img = cv2.cvtColor(yv12, cv2.COLOR_YUV2RGB_YV12)
    #     return img