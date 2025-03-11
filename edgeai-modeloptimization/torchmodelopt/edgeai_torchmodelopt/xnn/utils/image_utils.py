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

import copy
import cv2
import numpy as np


def chroma_blend(image, color, to_image_size=False):
    if image is None:
        return color
    elif color is None:
        return image
    #
    image_dtype = image.dtype
    color_dtype = color.dtype
    if image_dtype in (np.float32, np.float64):
        image = (image * 255).clip(0,255).astype(np.uint8)
    #
    if color_dtype in (np.float32, np.float64):
        color = (color * 255).clip(0,255).astype(np.uint8)
    #
    if image.shape != color.shape:
        if to_image_size:
            color = cv2.resize(color, dsize=(image.shape[1], image.shape[0]))
        else:
            image = cv2.resize(image, dsize=(color.shape[1], color.shape[0]))
        #
    #
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_y,image_u,image_v = cv2.split(image_yuv)
    color_yuv = cv2.cvtColor(color, cv2.COLOR_BGR2YUV)
    color_y,color_u,color_v = cv2.split(color_yuv)
    image_y = np.uint8(image_y)
    color_u = np.uint8(color_u)
    color_v = np.uint8(color_v)
    image_yuv = cv2.merge((image_y,color_u,color_v))
    image = cv2.cvtColor(image_yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
    if image_dtype in (np.float32, np.float64):
        image = image / 255.0
    #
    return image


def chroma_blend_alpha(image, color):
    alpha = 0.1
    image = alpha*image + (1-alpha)*color
    return image



# Author: Manu Mathew
# Date: 2021 March
def get_color_palette_generic(num_classes):
    num_classes_3 = np.power(num_classes, 1.0/3)
    delta_color = int(256/num_classes_3)
    colors = [(r, g, b) for r in range(0,256,delta_color)
                        for g in range(0,256,delta_color)
                        for b in range(0,256,delta_color)]
    # spread the colors list to num_classes
    color_step = len(colors) / num_classes
    colors_list = []
    to_idx = 0
    while len(colors_list) < num_classes:
        from_idx = round(color_step * to_idx)
        if from_idx < len(colors):
            colors_list.append(colors[from_idx])
        else:
            break
        #
        to_idx = to_idx + 1
    #
    shortage = num_classes-len(colors_list)
    if shortage > 0:
        colors_list += colors[-shortage:]
    #
    if len(colors_list) < 256:
        colors_list += [(255,255,255)] * (256-len(colors_list))
    #
    assert len(colors_list) == 256, f'incorrect length for color palette {len(colors_list)}'
    return colors_list


def get_color_palette(num_classes):
    if num_classes < 8:
        color_step = 255
    elif num_classes < 27:
        color_step = 127
    elif num_classes < 64:
        color_step = 63        
    else:
        color_step  = 31
    #
    color_map = [(r, g, b) for r in range(0, 256, color_step) for g in range(0, 256, color_step) for b in range(0, 256, color_step)]
    return color_map


def segmap_to_color(seg_img, num_classes):
    color_map = get_color_palette(num_classes)
    r = copy.deepcopy(seg_img)
    g = copy.deepcopy(seg_img)
    b = copy.deepcopy(seg_img)
    for l in range(0, num_classes):
        r[seg_img == l] = color_map[l][0]
        g[seg_img == l] = color_map[l][1]
        b[seg_img == l] = color_map[l][2]
    #
    rgb = np.zeros((seg_img.shape[0], seg_img.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb
