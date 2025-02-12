#!/usr/bin/env python

#################################################################################
# Copyright (c) 2018-2021, Texas Instruments Incorporated - http://www.ti.com
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
import os
import cv2
import argparse
import datetime
import numpy as np

################################
from edgeai_tensorvision.xengine import evaluate_pixel2pixel

################################
# to avoid hangs in data loader with multi threads
# this was observed after using cv2 image processing functions
# https://github.com/pytorch/pytorch/issues/1355
cv2.setNumThreads(0)

################################
#Create the parse and set default arguments
args = evaluate_pixel2pixel.get_config()

args.eval_semantic = True             # Set for 19 class cityscapes semantic segmentation. Works on native cityscapes dataset
args.eval_semantic_five_class = False   # Set for 5 class semantic segmentation
args.eval_motion = False               # Set for motion segmentation
args.eval_depth = False                # Set for depth estimation

################################
# Set arguments
args.label_path = '/user/a0132471/Files/pytorch/pytorch-devkit/data/datasets/cityscapes_768x384/data/gtFine/val'
                    # '/user/a0132471/Files/pytorch/pytorch-jacinto-models/data/datasets/tiad/data/tiad_08_03_2019/gtFine/val'
                    #'/user/a0132471/Files/pytorch/pytorch-jacinto-models/data/datasets/tiad/data/depth_release_0p8/val'
                    #'/user/a0132471/Files/pytorch/pytorch-jacinto-models/data/datasets/cityscapes_768x384/data/gtFine/val/'
                    #'/user/a0132471/Files/pytorch/pytorch-jacinto-models/data/datasets/tiad/data/gtFine/val'

args.infer_path =  '/user/a0132471/Files/c7x-mma-tidl/ti_dl/test/dump_cityscapes/calibrated/img'
                    #'/user/a0132471/Files/c7x-mma-tidl/ti_dl/test/dump_tiad/multi_task/release_0p8/semantic_bias_16_bil_fixed/img'
                    #'/user/a0132471/Files/c7x-mma-tidl/ti_dl/test/dump_tiad/tiad_semantic/img'
                    #'/user/a0132471/Files/pytorch/pytorch-jacinto-models/checkpoints/tiad_depth_semantic_motion_image_dof_conf_measure/2019-07-31-11-38-12_tiad_depth_semantic_motion_image_dof_conf_measure_deeplabv3plus_edgeailite_mobilenetv2_ericsun_mi4_resize768x384_benchmark_0p9/Task0'
                    #'/user/a0132471/Files/pytorch/pytorch-jacinto-models/checkpoints/cityscapes_segmentation_measure/2019-07-16-11-33-24_cityscapes_segmentation_measure_deeplabv3plus_edgeailite_mobilenetv2_tv_resize768x384/Task0_temp'
                    #'/user/a0132471/Files/c7x-mma-tidl/ti_dl/test/dump_cityscapes/tidl_cs'

args.output_size = (1024, 2048)          #(720, 1280) #(1024, 2048)
args.scale_factor = 1.0
args.verbose = True
args.inf_suffix = '.png'
args.depth = [False]
args.frame_IOU = False
################################
# Run the inference

evaluate_pixel2pixel.main(args)
