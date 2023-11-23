#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################

import math
import random
import numpy as np
import torch
import scipy
import warnings
import cv2
from ..layers import functional
from . import image_utils


###############################################################
# signed_log: a logarithmic representation with sign
def signed_log(x, base):
    def log_fn(x):
        return torch.log2(x)/np.log2(base)
    #
    # not using torch.sign as it doesn't have gradient
    sign = (x < 0) * (-1) + (x >= 0) * (+1)
    y = log_fn(torch.abs(x) + 1.0)
    y = y * sign
    return y


# convert back to linear from signed_log
def signed_pow(x, base):
    # not using torch.sign as it doesn't have gradient
    sign = (x < 0) * (-1) + (x >= 0) * (+1)
    y = torch.pow(base, torch.abs(x)) - 1.0
    y = y * sign
    return y


##################################################################
def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


###########################################################################
def tensor2img(tensor, adjust_range=True, min_value = None, max_value=None):
    if tensor.ndimension() < 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndimension() < 4:
        tensor = tensor.unsqueeze(0)
    if min_value is None:
        min_value = tensor.min()
    if max_value is None:
        max_value = tensor.max()
    range = max_value-min_value
    array = (255*(tensor - min_value)/range).clamp(0,255) if adjust_range else tensor
    if array.size(1) >= 3:
        img = torch.stack((array[0,0], array[0,1], array[0,2]), dim=2)
    else:
        img = array[0,0]
    return img.cpu().data.numpy().astype(np.uint8)


def flow2rgb(flow_map, max_value):
    global args
    _, h, w = flow_map.shape
    #flow_map[:,(flow_map[0] == 0) & (flow_map[1] == 0)] = float('nan')
    rgb_map = np.ones((h,w,3)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map / max_value
    else:
        normalized_flow_map = flow_map / (np.abs(flow_map).max())
    rgb_map[:,:,0] += normalized_flow_map[0]
    rgb_map[:,:,1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[:,:,2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)


def flow2hsv(flow_map, max_value=128, scale_fact=8, confidence=False):
    global args
    _, h, w = flow_map.shape
    hsv = np.zeros((h, w, 3)).astype(np.float32)

    mag = np.sqrt(flow_map[0]**2 + flow_map[1]**2)
    phase = np.arctan2(flow_map[1], flow_map[0])
    phase = np.mod(phase/(2*np.pi), 1)

    hsv[:, :, 0] = phase*360
    hsv[:, :, 1] = (mag*scale_fact/max_value).clip(0, 1)
    hsv[:, :, 2] = (scale_fact - hsv[:, :, 1]).clip(0, 1)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if confidence:
        return rgb * flow_map[2] > 128
    else:
        return rgb


def tensor2array(tensor, max_value=255.0, colormap='rainbow', input_blend=None):
    max_value = float(tensor.max()) if max_value is None else max_value

    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if cv2.__version__.startswith('2') :
                color_cvt = cv2.cv.CV_BGR2RGB
            else:  # 3.x,4,x
                color_cvt = cv2.COLOR_BGR2RGB
            #
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'magma': # >=3.4.8
                colormap = cv2.COLORMAP_MAGMA
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            elif colormap == 'plasma': # >=4.1
                colormap = cv2.COLORMAP_PLASMA
            elif colormap == 'turbo': # >=4.1.2
                colormap = cv2.COLORMAP_TURBO
            #
            array = (255.0*tensor.squeeze().numpy()/max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32) / 255.0
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            #
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)
    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy().transpose(1, 2, 0)*0.5
    #
    if input_blend is not None:
        array = image_utils.chroma_blend(input_blend, array)
    #
    return array


def tensor2img(tensor, max_value=63535):
    array = (63535*tensor.numpy()/max_value).clip(0, 63535).astype(np.uint16)
    if tensor.ndimension() == 3:
        assert (array.size(0) == 3)
        array = array.transpose(1, 2, 0)
    return array


##################################################################
def inverse_warp_flow(img, flow, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        flow: flow to be used for warping
    Returns:
        Source image warped to the target image plane
    """

    #check_sizes(img, 'img', 'B3HW')
    check_sizes(flow, 'flow', 'B2HW')

    b,c,h,w = img.size()
    h2 = (h-1.0)/2.0
    w2 = (w-1.0)/2.0

    pixel_coords = img_set_id_grid_(img)

    src_pixel_coords = pixel_coords + flow

    x_coords = src_pixel_coords[:, 0]
    x_coords = (x_coords - w2) / w2

    y_coords = src_pixel_coords[:, 1]
    y_coords = (y_coords - h2) / h2

    src_pixel_coords = torch.stack((x_coords, y_coords), dim=3)
    projected_img = torch.nn.functional.grid_sample(img, src_pixel_coords, \
                          mode='bilinear', padding_mode=padding_mode)

    return projected_img


def img_set_id_grid_(img):
    b, c, h, w = img.size()
    x_range = torch.Tensor(torch.arange(0, w).view(1, 1, w).expand(1,h,w)).type_as(img)  # [1, H, W]
    y_range = torch.Tensor(torch.arange(0, h).view(1, h, 1).expand(1,h,w)).type_as(img)  # [1, H, W]
    pixel_coords = torch.stack((x_range, y_range), dim=1).float()  # [1, 2, H, W]
    return pixel_coords


def crop_like(input, target):
    if target is None or (input.size()[2:] == target.size()[2:]):
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


def crop_alike(input, target):
    global crop_alike_warning_done
    if target is None or (input.size() == target.size()):
        return input, target

    warnings.warning('=> tensor dimension mismatch. input:{}, target:{}. cropping'.ormat(input.size(),target.size()))

    min_ch = min(input.size(1), target.size(1))
    min_h = min(input.size(2), target.size(2))
    min_w = min(input.size(3), target.size(3))
    h_offset_i = h_offset_t = w_offset_i = w_offset_t = 0
    if input.size(2) > target.size(2):
        h_offset_i = (input.size(2) - target.size(2))//2
    else:
        h_offset_t = (target.size(2) - input.size(2))//2

    if input.size(3) > target.size(3):
        w_offset_i = (input.size(3) - target.size(3))//2
    else:
        w_offset_t = (target.size(3) - input.size(3))//2

    input = input[:, :min_ch, h_offset_i:(h_offset_i+min_h), w_offset_i:(w_offset_i+min_w)]
    target = target[:, :min_ch, h_offset_t:(h_offset_t+min_h), w_offset_t:(w_offset_t+min_w)]

    return input, target


def align_channels_(x,y):
    chan_x = x.size(1)
    chan_y = y.size(1)
    if chan_x != chan_y:
        chan_min = min(chan_x, chan_y)
        x = x[:,:chan_min,...]
        if len(x.size()) < 4:
            x = torch.unsqueeze(x,dim=1)
        y = y[:,:chan_min,...]
        if len(y.size()) < 4:
            y = torch.unsqueeze(y,dim=1)
    return x, y


def debug_dump_tensor(tensor, image_name, adjust_range=True):
    img = tensor2img(tensor, adjust_range=adjust_range)
    scipy.misc.imsave(image_name, img)



