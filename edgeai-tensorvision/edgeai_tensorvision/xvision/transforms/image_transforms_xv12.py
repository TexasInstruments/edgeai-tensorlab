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
from .image_transforms import *


def yv12_to_nv12_image(yv12=None):
    image_w = yv12.shape[1]
    image_h = (yv12.shape[0] * 2) // 3
    y_h = image_h
    uv_h = image_h // 4
    u_w = image_w // 2

    Y = yv12[0:image_h, 0:image_w]
    V = yv12[y_h:y_h + uv_h, 0:image_w]
    U = yv12[y_h + uv_h:y_h + 2 * uv_h, 0:image_w]

    UV = np.zeros((Y.shape[0] // 2, Y.shape[1]), dtype='uint8')

    # U00V00   U01V01 ....
    # U10V10   U11V11 ....
    # U20V20   U21V21 ....

    UV[0::2, 0::2] = U[:, 0:u_w]
    UV[0::2, 1::2] = V[:, 0:u_w]

    UV[1::2, 0::2] = U[:, u_w:]
    UV[1::2, 1::2] = V[:, u_w:]
    Y = np.expand_dims(Y, axis=2)
    UV = np.expand_dims(UV, axis=2)

    img = [Y, UV]

    test = False
    if test:
        op_yuv = np.zeros((yv12.shape))
        U[:, 0:u_w] = UV[0::2, 0::2, 0]
        V[:, 0:u_w] = UV[0::2, 1::2, 0]
        U[:, u_w:] = UV[1::2, 0::2, 0]
        V[:, u_w:] = UV[1::2, 1::2, 0]
        op_yuv[0:image_h, 0:image_w] = Y[:, :, 0]
        op_yuv[y_h:y_h + uv_h, 0:image_w] = V
        op_yuv[y_h + uv_h:y_h + 2 * uv_h, 0:image_w] = U
        assert (np.array_equal(yv12, op_yuv))

    return img

def nv12_to_bgr_image(Y = None, UV = None, image_scale = None, image_mean = None):
    image_h = Y.shape[1]
    image_w = Y.shape[2]

    y_h = image_h
    uv_h = image_h // 4
    u_w = image_w // 2

    op_yuv = torch.zeros(((image_h*3)//2, image_w), device=Y.device)
    U = torch.zeros((uv_h, image_w), device=Y.device)
    V = torch.zeros((uv_h, image_w), device=Y.device)
    U[:, 0:u_w] = UV[0, 0::2, 0::2]
    V[:, 0:u_w] = UV[0, 0::2, 1::2]
    U[:, u_w:] = UV[0, 1::2, 0::2]
    V[:, u_w:] = UV[0, 1::2, 1::2]
    op_yuv[0:image_h, 0:image_w] = Y[0, :, :]
    op_yuv[y_h:y_h + uv_h, 0:image_w] = V
    op_yuv[y_h + uv_h:y_h + 2 * uv_h, 0:image_w] = U

    op_yuv = op_yuv / torch.tensor(image_scale, device=Y.device) + torch.tensor(image_mean, device=Y.device)
    op_yuv = op_yuv.cpu().numpy().astype('uint8')
    bgr = cv2.cvtColor(op_yuv, cv2.COLOR_YUV2BGR_YV12)
    return bgr


class RGBtoYV12(object):
    def __init__(self, is_flow=None, keep_rgb=False):
        self.is_flow = is_flow
        self.keep_rgb = keep_rgb


    def debug_print(self, img=None, enable=False):
        if enable:
            h = img.shape[0] * 2 // 3
            w = img.shape[1]

            print("-" * 32, "RGBtoYV12")
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

    def __call__(self, images, target):
        for img_idx in range(len(images)):
            is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
            if not is_flow_img:
                images[img_idx] = cv2.cvtColor(images[img_idx], cv2.COLOR_RGB2YUV_YV12)
                self.debug_print(img = images[img_idx], enable=False)
        return images, target


#convert from YV12 to YUV444 with optional resize
def yv12_to_yuv444(img=None, out_size=None):
    in_w = img.shape[1]
    in_h = (img.shape[0] * 2) // 3

    y_h = in_h
    in_uv_h = in_h // 4

    Y = img[0:in_h, 0:in_w]

    V = img[y_h:y_h + in_uv_h, 0:in_w]
    V = V.reshape(V.shape[0]*2, -1)
    
    U = img[y_h + in_uv_h:y_h + 2 * in_uv_h, 0:in_w]
    U = U.reshape(U.shape[0] * 2, -1)

    #if op_size is none then use input size as outsize In that case this functio becomes YV12 to YUV444
    if out_size is None:
        out_size = (in_h, in_w)

    out_h, out_w = out_size

    Y = cv2.resize(Y, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    U = cv2.resize(U, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    V = cv2.resize(V, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    yuv444 = np.zeros((in_h, in_w, 3), dtype=np.uint8)
    yuv444[:, :, 0] = Y
    yuv444[:, :, 1] = U
    yuv444[:, :, 2] = V
    return yuv444

def get_w_b_yuv_to_rgb(device=None, rounding=True):
    offset = OFFSET if rounding else 0

    uv_mean = 128
    w_yuv_to_rgb = torch.tensor([ITUR_BT_601_CY, 0,              ITUR_BT_601_CVR,
                                ITUR_BT_601_CY, ITUR_BT_601_CUG, ITUR_BT_601_CVG,
                                ITUR_BT_601_CY, ITUR_BT_601_CUB, 0], dtype=torch.float, device=device).reshape(3,3)

    #print(w_yuv_to_rgb)
    w_yuv_to_rgb = w_yuv_to_rgb/(1<<ITUR_BT_601_SHIFT)
    #print("w_yuv_to_rgb: ")
    #print(w_yuv_to_rgb)

    b_yuv_to_rgb = torch.tensor([offset-ITUR_BT_601_CVR*uv_mean-16*ITUR_BT_601_CY,
                                 offset-ITUR_BT_601_CVG*uv_mean-ITUR_BT_601_CUG*uv_mean-16*ITUR_BT_601_CY,
                                 offset-ITUR_BT_601_CUB*uv_mean-16*ITUR_BT_601_CY], dtype=torch.float, device=device).reshape(3,1)
    #print(b_yuv_to_rgb)
    b_yuv_to_rgb = b_yuv_to_rgb/(1<<ITUR_BT_601_SHIFT)
    #print("b_yuv_to_rgb: ")
    #print(b_yuv_to_rgb)

    if device == 'cpu':
        w_yuv_to_rgb = w_yuv_to_rgb.cpu().numpy()
        b_yuv_to_rgb = b_yuv_to_rgb.cpu().numpy()

    return [w_yuv_to_rgb, b_yuv_to_rgb]


#r = ((y-16) * ITUR_BT_601_CY + OFFSET + ITUR_BT_601_CVR * (v-uv_mean) ) >> ITUR_BT_601_SHIFT
#g = ((y-16) * ITUR_BT_601_CY + OFFSET + ITUR_BT_601_CVG * (v-uv_mean) ) + ITUR_BT_601_CUG * (u-uv_mean) ) >> ITUR_BT_601_SHIFT
#b = ((y-16) * ITUR_BT_601_CY + OFFSET + ITUR_BT_601_CUB * (u-uv_mean) ) >> ITUR_BT_601_SHIFT

#r = (y * ITUR_BT_601_CY + ITUR_BT_601_CVR * v + (OFFSET-ITUR_BT_601_CVR*uv_mean -16*ITUR_BT_601_CY) ) >> ITUR_BT_601_SHIFT
#g = (y * ITUR_BT_601_CY + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u + (OFFSET- ITUR_BT_601_CVG*uv_mean-ITUR_BT_601_CUG*uv_mean-16*ITUR_BT_601_CY)) >> ITUR_BT_601_SHIFT
#b = (y * ITUR_BT_601_CY + ITUR_BT_601_CUB * u + (OFFSET-ITUR_BT_601_CUB*uv_mean-16*ITUR_BT_601_CY) ) >> ITUR_BT_601_SHIFT

# w_yuv_to_rgb = np.array([ITUR_BT_601_CY, 0,               ITUR_BT_601_CVR,
#                          ITUR_BT_601_CY, ITUR_BT_601_CUG, ITUR_BT_601_CVG,
#                          ITUR_BT_601_CY, ITUR_BT_601_CUB, 0]).reshape(3,3)
#
# b_yuv_to_rgb = np.array([OFFSET-ITUR_BT_601_CVR*uv_mean-16*ITUR_BT_601_CY,
#                          OFFSET-ITUR_BT_601_CVG*uv_mean-ITUR_BT_601_CUG*uv_mean-16*ITUR_BT_601_CY,
#                          OFFSET-ITUR_BT_601_CUB*uv_mean-16*ITUR_BT_601_CY]).reshape(3,1)

#ref https://github.com/opencv/opencv/blob/8c0b0714e76efef4a8ca2a7c410c60e55c5e9829/modules/imgproc/src/color_yuv.simd.hpp#L1075
ITUR_BT_601_CY = 1220542
ITUR_BT_601_CUB = 2116026
ITUR_BT_601_CUG = -409993
ITUR_BT_601_CVG = -852492
ITUR_BT_601_CVR = 1673527
ITUR_BT_601_SHIFT = 20
OFFSET = (1 << (ITUR_BT_601_SHIFT - 1))


def report_stat(diff=None):
    unique, counts = np.unique(diff, return_counts=True)
    result = dict(zip(unique, counts))
    print(result)
    counts_list = np.zeros(max(unique) + 1, dtype=np.int)
    for (diff, count) in zip(unique, counts):
        counts_list[diff] = count

    str_to_print = ','.join('%d' % x for x in counts_list)
    print(str_to_print)
    counts_list = (100.00 * counts_list) / sum(counts_list)
    str_to_print = ','.join('%8.5f' % x for x in counts_list)
    print(str_to_print)

def compare_diff(tensor_ref=None, tensor=None, exact_comp=False, roi_h=None, roi_w=None, ch_axis_idx=2, auto_scale=False):
    roi_h = [0,tensor.shape[1]] if roi_h is None else roi_h
    roi_w = [0,tensor.shape[2]] if roi_w is None else roi_w

    if ch_axis_idx == 0:
        # swap ch index as subsequent code assumes ch_idx to be 2
        tensor = np.moveaxis(tensor, 0,-1)
        tensor_ref = np.moveaxis(tensor_ref, 0, -1)

    n_ch = tensor.shape[2]
    if ch_axis_idx != 0 and ch_axis_idx != 2:
        exit("wrong ch index in compare_diff()")

    # crop
    tensor = tensor[roi_h[0]:roi_h[1], roi_w[0]:roi_w[1], :]
    tensor_ref = tensor_ref[roi_h[0]:roi_h[1], roi_w[0]:roi_w[1], :]

    #needed for float arrays. Convert float arrays to int with range [-128, 128]
    if auto_scale:
        max_val = np.amax(np.abs(tensor_ref))
        scale = 128.0/max_val
        tensor_ref = (tensor_ref * scale).astype(np.int)
        tensor = (tensor * scale).astype(np.int)

    if exact_comp:
        for ch in range(n_ch):
            print("ch: ", ch, " matching: ", np.array_equal(tensor_ref[:, :, ch], tensor[:, :, ch]))
            indices = np.where(tensor_ref[:, :, ch] != tensor[:, :, ch])
            for (idx_h, idx_w) in zip(indices[0],indices[1]):
                print(tensor_ref[idx_h, idx_w, ch], " : ", tensor[idx_h, idx_w, ch])
    else: #if clip is not used it is not expected to match with ref so just find how many are differing
        print(" Global stats:")
        diff = np.abs(tensor_ref[:, :, :] - tensor[:, :, :])
        report_stat(diff)
        for ch in range(n_ch):
            print(" ========= ch:", ch)
            diff = np.abs(tensor_ref[:, :, ch] - tensor[:, :, ch])
            report_stat(diff)


def image_padding(img=None, pad_vals=[0,0,0]):
    img_padded = np.empty((img.shape[0] + 2, img.shape[1] + 2, img.shape[2]), dtype=img.dtype)
    img_padded[:, :, 0] = pad_vals[0]
    img_padded[:, :, 1] = pad_vals[1]
    img_padded[:, :, 2] = pad_vals[2]
    img_padded[1:-1, 1:-1, :] = img[:, :, :]
    return img_padded


def opencv_yuv_to_rgb(yuv444=None, uv_mean=0, clip=False, rounding=True, matrix_based_implementation=False, 
    op_type=np.int):

    if matrix_based_implementation:
        implementation_with_loops = False
        rgb = np.empty_like(yuv444, dtype=op_type)
        [w_yuv_to_rgb, b_yuv_to_rgb] = get_w_b_yuv_to_rgb(device='cpu', rounding=rounding)

        if implementation_with_loops:
            for y in range(yuv444.shape[0]):
                for x in range(yuv444.shape[1]):
                    yuv444_sample = yuv444[y,x,:].reshape(3,1).astype(np.float)
                    temp = w_yuv_to_rgb @ yuv444_sample + b_yuv_to_rgb.reshape(3,1)
                    rgb[y,x,:] = np.squeeze(temp)
        else:
            r = np.dot(yuv444, w_yuv_to_rgb[0]) + b_yuv_to_rgb[0]
            g = np.dot(yuv444, w_yuv_to_rgb[1]) + b_yuv_to_rgb[1]
            b = np.dot(yuv444, w_yuv_to_rgb[2]) + b_yuv_to_rgb[2]
            rgb[:, :, 0] = r
            rgb[:, :, 1] = g
            rgb[:, :, 2] = b
        #compare_diff(tensor_ref=rgb_matrix_based, tensor=rgb, exact_comp=True, ch_axis_idx=2)
    else:
        y = yuv444[:, :, 0].astype(dtype=np.int) - 16
        u = yuv444[:, :, 1].astype(dtype=np.int) - uv_mean
        v = yuv444[:, :, 2].astype(dtype=np.int) - uv_mean
        ruv = OFFSET + ITUR_BT_601_CVR * v
        guv = OFFSET + ITUR_BT_601_CVG * v + ITUR_BT_601_CUG * u
        buv = OFFSET + ITUR_BT_601_CUB * u

        y00 = np.maximum(y, 0) * ITUR_BT_601_CY if clip else y * ITUR_BT_601_CY

        r = (y00 + ruv) >> ITUR_BT_601_SHIFT
        g = (y00 + guv) >> ITUR_BT_601_SHIFT
        b = (y00 + buv) >> ITUR_BT_601_SHIFT

        if clip:
            r = np.clip(r, 0, 255)
            g = np.clip(g, 0, 255)
            b = np.clip(b, 0, 255)

        rgb = np.empty(yuv444.shape, dtype=np.int)
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b

    #compare_diff(rgb_ref=rgb, rgb=rgb_matrix_based, clip=False)
    return rgb


class YV12toRGB(object):
    def __init__(self, is_flow=None, keep_rgb=False):
        self.is_flow = is_flow
        self.keep_rgb = keep_rgb

    def __call__(self, images, target):
        for img_idx in range(len(images)):
            is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
            if not is_flow_img:
                images[img_idx] = cv2.cvtColor(images[img_idx], cv2.COLOR_YUV2RGB_YV12)
        return images, target

#similar to YV12toRGB but without clip
class YV12toRGBWithoutClip(object):
    def __init__(self, is_flow=None, keep_rgb=False):
        self.is_flow = is_flow
        self.keep_rgb = keep_rgb

    def __call__(self, images, target):
        for img_idx in range(len(images)):
            is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
            if not is_flow_img:
                yuv444 = yv12_to_yuv444(images[img_idx])
                images[img_idx] = opencv_yuv_to_rgb(yuv444=yuv444, uv_mean=128, matrix_based_implementation=False)
        return images, target


# Padding around images boundaries  
class ImagePadding(object):
    def __init__(self, is_flow=None, keep_rgb=False, pad_vals=[0,0,0]):
        self.is_flow = is_flow
        self.keep_rgb = keep_rgb
        self.pad_vals = pad_vals

    def __call__(self, images, target):
        for img_idx in range(len(images)):
            is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
            if not is_flow_img:
                images[img_idx] = image_padding(img=images[img_idx], pad_vals=self.pad_vals)
        return images, target

#YV12 to YUV444 
class YV12toYUV444(object):
    def __init__(self, is_flow=None, keep_rgb=False):
        self.is_flow = is_flow
        self.keep_rgb = keep_rgb

    def __call__(self, images, target):
        for img_idx in range(len(images)):
            is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
            if not is_flow_img:
                images[img_idx] = yv12_to_yuv444(images[img_idx])
        return images, target

class YV12toNV12(object):
    def __init__(self, is_flow=None, keep_rgb=False):
        self.is_flow = is_flow
        self.keep_rgb = keep_rgb

    def __call__(self, images, target):
        for img_idx in range(len(images)):
            [Y, UV] = yv12_to_nv12_image(yv12=images[img_idx])
            rgb = cv2.cvtColor(images[img_idx], cv2.COLOR_YUV2RGB_YV12)
            images[img_idx] = [Y, UV, rgb] if self.keep_rgb else [Y, UV]

        return images, target


class RGBtoNV12(object):
    def __init__(self, is_flow=None, keep_rgb=False):
        self.is_flow = is_flow
        self.keep_rgb = keep_rgb

    def __call__(self, images, target):

        for img_idx in range(len(images)):
            is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
            if not is_flow_img:
                # current
                yv12 = cv2.cvtColor(images[img_idx], cv2.COLOR_RGB2YUV_YV12)

                # cfg:20
                # yv12 = cv2.cvtColor(images[img_idx], cv2.COLOR_BGR2YUV_YV12)
                [Y, UV] = yv12_to_nv12_image(yv12=yv12)
                images[img_idx] = [Y, UV, images[img_idx]] if self.keep_rgb else [Y, UV]
        return images, target


class RGBtoNV12toRGB(object):
    def __init__(self, is_flow=None):
        self.is_flow = is_flow

    def __call__(self, images, target):

        for img_idx in range(len(images)):
            is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
            if not is_flow_img:
                # OpenCV does not support cv2.COLOR_RGB2YUV_NV12 so instead use YV12 as
                # intermediate format. Final effect should be same.
                yuv = cv2.cvtColor(images[img_idx], cv2.COLOR_RGB2YUV_YV12)
                images[img_idx] = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_YV12)
        return images, target


class RandomScaleCropYV12(RandomScaleCrop):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __init__(self, img_resize, scale_range=(1.0, 2.0), is_flow=None, center_crop=False):
        super().__init__(img_resize, scale_range=scale_range, is_flow=is_flow, center_crop=center_crop, resize_in_yv12=True)


class RandomCropScaleYV12(object):
    """Crop the Image to random size and scale to the given resolution"""

    def __init__(self, size, crop_range=(0.08, 1.0), is_flow=None, center_crop=False):
        super().__init__(self, size, crop_range=crop_range, is_flow=is_flow, center_crop=center_crop, resize_in_yv12=True)


class ScaleYV12(object):
    def __init__(self, img_size, target_size=None, is_flow=None):
        self.img_size = img_size
        self.target_size = target_size if target_size else img_size
        self.is_flow = is_flow

    def __call__(self, images, targets):
        if self.img_size is None:
            return images, targets

        def func_img(img, img_idx):
            is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
            img = ImageTransformUtils.resize_img_yv12(img, self.img_size, interpolation=-1, is_flow=is_flow_img)
            return img

        def func_tgt(img, img_idx):
            is_flow_tgt = (self.is_flow[1][img_idx] if self.is_flow else self.is_flow)
            img = ImageTransformUtils.resize_img(img, self.target_size, interpolation=cv2.INTER_NEAREST,
                                                 is_flow=is_flow_tgt)
            return img

        images = ImageTransformUtils.apply_to_list(func_img, images)
        targets = ImageTransformUtils.apply_to_list(func_tgt, targets)

        return images, targets



class ScaleYV12(object):
    def __init__(self, img_size, target_size=None, is_flow=None):
        self.img_size = img_size
        self.target_size = target_size if target_size else img_size
        self.is_flow = is_flow

    def __call__(self, images, targets):
        if self.img_size is None:
            return images, targets

        def func_img(img, img_idx):
            is_flow_img = (self.is_flow[0][img_idx] if self.is_flow else self.is_flow)
            img = ImageTransformUtils.resize_img_yv12(img, self.img_size, interpolation=-1, is_flow=is_flow_img)
            return img

        def func_tgt(img, img_idx):
            is_flow_tgt = (self.is_flow[1][img_idx] if self.is_flow else self.is_flow)
            img = ImageTransformUtils.resize_img(img, self.target_size, interpolation=cv2.INTER_NEAREST,
                                                 is_flow=is_flow_tgt)
            return img

        images = ImageTransformUtils.apply_to_list(func_img, images)
        targets = ImageTransformUtils.apply_to_list(func_tgt, targets)

        return images, targets

