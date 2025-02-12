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
from edgeai_torchmodelopt.xnn.utils import str2bool
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default=None, help='checkpoint save folder')
parser.add_argument('--gpus', type=int, nargs='*', default=None, help='Base learning rate')
parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
# parser.add_argument('--lr', type=float, default=None, help='Base learning rate')
# parser.add_argument('--lr_clips', type=float, default=None, help='Learning rate for clips in PAct2')
parser.add_argument('--lr_calib', type=float, default=None, help='Learning rate for calibration')
parser.add_argument('--model_name', type=str, default=None, help='model name')
parser.add_argument('--dataset_name', type=str, default=None, help='dataset name')
parser.add_argument('--data_path', type=str, default=None, help='data path')
parser.add_argument('--epoch_size', type=float, default=None, help='epoch size. using a fraction will reduce the data used for one epoch')
# parser.add_argument('--epochs', type=int, default=None, help='number of epochs')
# parser.add_argument('--milestones', type=int, nargs=2, default=None, help='change lr at these milestones')
parser.add_argument('--img_resize', type=int, default=None, help='images will be first resized to this size during training and validation')
# parser.add_argument('--rand_scale', type=float, nargs=2, default=None, help='during training (only) fraction of the image to crop (this will then be resized to img_crop)')
parser.add_argument('--img_crop', type=int, default=None, help='the cropped portion (validation), cropped pertion will be resized to this size (training)')
parser.add_argument("--quantization", "--quantize", dest="quantize", type=str2bool, default=None, help='Quantize the model')
#parser.add_argument('--model_surgery', type=str, default=None, choices=[None, 'pact2'], help='whether to transform the model after defining')
parser.add_argument('--pretrained', type=str, default=None, help='pretrained model')
# parser.add_argument('--resume', type=str, default=None, help='resume an unfinished training from this model')
parser.add_argument('--bitwidth_weights', type=int, default=None, help='bitwidth for weight quantization')
parser.add_argument('--bitwidth_activations', type=int, default=None, help='bitwidth for activation quantization')
parser.add_argument('--constrain_bias', type=int, default=None, help='constrain_bias')
cmds = parser.parse_args()

################################
# taken care first, since this has to be done before importing pytorch
if 'gpus' in vars(cmds):
    value = getattr(cmds, 'gpus')
    if (value is not None) and ("CUDA_VISIBLE_DEVICES" not in os.environ):
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(v) for v in value])
    #
#

# to avoid hangs in data loader with multi threads
# this was observed after using cv2 image processing functions
# https://github.com/pytorch/pytorch/issues/1355
cv2.setNumThreads(0)

################################
from edgeai_tensorvision.xengine import test_classification

#Create the parse and set default arguments
args = test_classification.get_config()

################################
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

################################
#Set arguments
args.model_name = 'mobilenetv2_tv_x1' #'resnet50_x1', 'mobilenetv2_tv_x1', 'mobilenetv2_ericsun_x1', 'mobilenetv2_shicai_x1'

args.dataset_name = 'image_folder_classification' # 'image_folder_classification', 'imagenet_classification', 'cifar10_classification', 'cifar100_classification'

#args.save_path = './data/checkpoints/edgeailite'

args.data_path = f'./data/datasets/{args.dataset_name}'

args.pretrained = 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
                #'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
                #'./data/modelzoo/pretrained/pytorch/imagenet_classification/torchvision/resnet50-19c8e357.pth'
                #'./data/modelzoo/pretrained/pytorch/imagenet_classification/ericsun99/MobileNet-V2-Pytorch/mobilenetv2_Top1_71.806_Top2_90.410.pth.tar'
                #'./data/modelzoo/pretrained/pytorch/imagenet_classification/shicai/MobileNet-Caffe/mobilenetv2_shicai_rgb.tar'



args.model_config.input_channels = 3
args.model_config.output_type = 'classification'
args.model_config.output_channels = None

args.batch_size = 64 #1 #256                   #16 #32 #64
args.workers    = 8 #1
#args.shuffle = True
#args.epoch_size = 0
args.count_flops = True

# args.quantize = True
# args.write_layer_ip_op = True

# args.histogram_range = True
# args.bias_calibration = True
# args.per_channel_q = False

args.phase = 'validation'
args.print_freq = 10 #100

################################
for key in vars(cmds):
    if key == 'gpus':
        pass # already taken care above, since this has to be done before importing pytorch
    elif hasattr(args, key):
        value = getattr(cmds, key)
        if value != 'None' and value is not None:
            setattr(args, key, value)
    else:
        assert False, f'invalid argument {key}'
#

################################
# these dependent on the dataset chosen
args.img_resize = (args.img_resize if args.img_resize else 256)
args.img_crop = (args.img_crop if args.img_crop else 224)
args.model_config.num_classes = (100 if 'cifar100' in args.dataset_name else (10  if 'cifar10' in args.dataset_name else 1000))
args.model_config.strides = (1,1,1,2,2) if args.img_crop<64 else ((1,1,2,2,2) if args.img_crop<128 else (2,2,2,2,2))


################################
#Run the training
test_classification.main(args)

