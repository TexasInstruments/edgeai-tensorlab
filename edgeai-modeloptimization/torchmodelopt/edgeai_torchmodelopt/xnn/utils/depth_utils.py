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

import sys
import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
import math

# FIX_ME:SN move to utils
def fish_cord_to_rect(dx_f=0, dy_f=0, theta_to_r_rect=[], params=[]):
    f_x = params.cam_info_K[0]
    f = f_x
    r_f = np.sqrt(dx_f * dx_f + dy_f * dy_f)
    calib_idx = int((r_f - params.fisheye_r_start) / params.fisheye_r_step)
    calib_idx = max(0, min(calib_idx, len(theta_to_r_rect) - 2))
    # getting theta using interpolation from theta_to_r_rect and then r
    r_f_lower = params.fisheye_r_start + calib_idx * params.fisheye_r_step
    theta = theta_to_r_rect[calib_idx] + (theta_to_r_rect[calib_idx + 1] - theta_to_r_rect[calib_idx]) * (
            r_f - r_f_lower) / params.fisheye_r_step

    r_rect = f * abs(math.tan(theta * np.pi / 180.0))
    if (r_f != 0.0):
        dx_r = dx_f * r_rect / r_f
        dy_r = dy_f * r_rect / r_f
    else:
        dx_r = 0.0
        dy_r = 0.0

    return dx_r, dy_r


def ZtoAbsYForaPixel(Zc=0.0, y_f=0.0, x_f=0.0, params=[]):
    cx_f = params.cam_info_K[2]
    cy_f = params.cam_info_K[5]
    f_x = params.cam_info_K[0]
    f_y = params.cam_info_K[4]

    dy_f = y_f - cy_f
    dx_f = x_f - cx_f

    dx_r, dy_r = fish_cord_to_rect(dx_f=dx_f, dy_f=dy_f, theta_to_r_rect=r_fish_to_theta_rect, params=params)
    Yc = dy_r * Zc / f_y
    Xc = dx_r * Zc / f_x
    M_c_l_r = np.array([[0.0046, 1.0000, -0.0061],
                        [0.45353, -0.0075, -0.8912],
                        [-0.8913, 0.0014, -0.4535]])  # Rotation matrix for TIAD_2017_seq1

    M_c_l_t = np.array([[0.6872, 0.2081, -2.2312]])
    M_c_l = np.vstack((np.hstack((M_c_l_r, M_c_l_t.transpose())), np.array([0, 0, 0, 1])))
    M_l_c = np.linalg.inv(M_c_l)
    print('Yc', Yc)
    [Xl, Yl, Zl, _] = np.matmul(M_l_c, np.array([Xc, Yc, Zc, 1]))
    print('Yl', Yl)
    return abs(Yl)


# This function converts Z to Y using fisheye model  + pinhole camera geometry
def ZtoY(image_Z=None):
    class Params:
        def __init__(self):
            # camera params

            self.cam_info_K = np.array([311.8333, 0.0000, 640.0000,
                                        0.0000, 311.8333, 360.0000,
                                        0.0000, 0.0000, 1.0000])
            self.fisheye_r_start = 0.0
            self.fisheye_r_step = 0.5

    params = Params()
    image_Y = image_Z
    height, width = image_Z.shape

    for row in range(height):
        for col in range(width):
            image_Y[row, col] = ZtoAbsYForaPixel(Zc=image_Z[row, col], y_f=row, x_f=col, params=params)
    return image_Y
