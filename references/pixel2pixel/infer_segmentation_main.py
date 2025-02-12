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
parser.add_argument('--model_name', type=str, default=None, help='model name')
parser.add_argument('--dataset_name', type=str, default=None, help='dataset name')
parser.add_argument('--data_path', type=str, default=None, help='data path')
parser.add_argument('--epoch_size', type=float, default=None, help='epoch size. using a fraction will reduce the data used for one epoch')
parser.add_argument('--img_resize', type=int, nargs=2, default=None, help='img_resize size. for training this will be modified according to rand_scale')
parser.add_argument('--rand_scale', type=float, nargs=2, default=None, help='random scale factors for training')
parser.add_argument('--rand_crop', type=int, nargs=2, default=None, help='random crop for training')
parser.add_argument('--output_size', type=int, nargs=2, default=None, help='output size of the evaluation - prediction/groundtruth. this is not used while training as it blows up memory requirement')
parser.add_argument("--quantization", "--quantize", dest="quantize", type=str2bool, default=None, help='Quantize the model')
#parser.add_argument('--model_surgery', type=str, default=None, choices=[None, 'pact2'], help='whether to transform the model after defining')
parser.add_argument('--pretrained', type=str, default=None, help='pretrained model')
parser.add_argument('--bitwidth_weights', type=int, default=None, help='bitwidth for weight quantization')
parser.add_argument('--bitwidth_activations', type=int, default=None, help='bitwidth for activation quantization')
parser.add_argument('--img_border_crop', type=int, nargs=4, default=None, help='image border crop rectangle. can be relative or absolute')
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
#import of torch should be after CUDA_VISIBLE_DEVICES for it to take effect
import torch
from edgeai_tensorvision.xengine import infer_pixel2pixel

# Create the parse and set default arguments
args = infer_pixel2pixel.get_config()


#Modify arguments
args.model_name = "deeplabv3plus_mobilenetv2_tv_edgeailite"

args.dataset_name = 'cityscapes_segmentation_measure' #'tiad_segmentation_infer'   #'cityscapes_segmentation_infer' #'tiad_segmentation'  #'cityscapes_segmentation_measure'
args.dataset_config.split = 'val'

#args.save_path = './data/checkpoints/edgeailite'
args.data_path = './data/datasets/cityscapes/data' #'./data/datasets/cityscapes/data'   #'/data/hdd/datasets/cityscapes_leftImg8bit_sequence_trainvaltest/' #'./data/datasets/cityscapes/data'  #'./data/tiad/data/demoVideo/sequence0021'  #'./data/tiad/data/demoVideo/sequence0025'   #'./data/tiad/data/demoVideo/sequence0001_2017'
#args.pretrained = './data/modelzoo/semantic_segmentation/cityscapes/deeplabv3plus_edgeailite-mobilenetv2/cityscapes_segmentation_deeplabv3plus_edgeailite-mobilenetv2_2019-06-26-08-59-32.pth'
#args.pretrained = './data/checkpoints/edgeailite/tiad_segmentation/2019-10-18_00-50-03_tiad_segmentation_deeplabv3plus_edgeailite_mobilenetv2_ericsun_resize768x384_traincrop768x384_float/checkpoint.pth.tar'

args.pretrained = '../edgeai-modelzoo/models/vision/segmentation/cityscapes/edgeai-jai/deeplabv3plus_edgeailite_mobilenetv2_768x384_20190626_checkpoint.pth'

args.model_config.input_channels = (3,)
args.model_config.output_type = ['segmentation']
args.model_config.output_channels = None
args.losses = [['segmentation_loss']]
args.metrics = [['segmentation_metrics']]

args.frame_IOU =  False # Print mIOU for each frame
args.shuffle = False

args.num_images = 50000   # Max number of images to run inference on

#['color'], ['blend'], ['']
args.viz_op_type = ['blend']
args.visualize_gt = False
args.car_mask = False  # False   #True
args.label = [True]    # False   #True
args.label_infer = [True]
args.palette = True
args.start_img_index = 0
args.end_img_index = 0
args.create_video = True  # True   #False
args.depth = [False]

args.epoch_size = 0                     #0 #0.5
args.iter_size = 1                      #2

args.batch_size = 32 #80                  #12 #16 #32 #64
args.img_resize = (384, 768)         #(256,512) #(512,512) # #(1024, 2048) #(512,1024)  #(720, 1280)

args.output_size = (1024, 2048)          #(1024, 2048)
#args.rand_scale = (1.0, 2.0)            #(1.0,2.0) #(1.0,1.5) #(1.0,1.25)

args.depth = [False]
args.quantize = False
args.histogram_range = True

#args.image_prenorm=False
#args.image_mean = [0]
#args.image_scale = [1.0]
#args.image_prenorm = False
#args.image_mean = [123.675, 116.28, 103.53]
#args.image_scale = [0.017125, 0.017507, 0.017429]
#args.image_mean = [0]                # image mean for input image normalization
#args.image_scale = [1.0]             # image scaling/mult for input iamge normalization

#save modified files after last commit
#args.save_mod_files = True

args.gpu_mode = True
args.write_layer_ip_op = False
args.save_onnx = True

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
#Run the test
infer_pixel2pixel.main(args)

