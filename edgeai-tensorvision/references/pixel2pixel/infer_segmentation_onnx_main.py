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

################################
#sys.path.insert(0, os.path.abspath('./modules'))


################################
from edgeai_torchmodelopt.xnn.utils import str2bool
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default=None, help='checkpoint save folder')
parser.add_argument('--gpus', type=int, nargs='*', default=None, help='Base learning rate')
parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
parser.add_argument('--lr', type=float, default=None, help='Base learning rate')
parser.add_argument('--model_name', type=str, default=None, help='model name')
parser.add_argument('--dataset', type=str, default=None, help='dataset name')
parser.add_argument('--data_path', type=str, default=None, help='data path')
parser.add_argument('--epoch_size', type=float, default=None, help='epoch size. using a fraction will reduce the data used for one epoch')
parser.add_argument('--epochs', type=int, default=None, help='number of epochs')
parser.add_argument('--milestones', type=int, nargs=2, default=None, help='change lr at these milestones')
parser.add_argument('--img_resize', type=int, nargs=2, default=None, help='img_resize size. for training this will be modified according to rand_scale')
parser.add_argument('--rand_scale', type=float, nargs=2, default=None, help='random scale factors for training')
parser.add_argument('--rand_crop', type=int, nargs=2, default=None, help='random crop for training')
parser.add_argument('--output_size', type=int, nargs=2, default=None, help='output size of the evaluation - prediction/groundtruth. this is not used while training as it blows up memory requirement')
parser.add_argument("--quantization", "--quantize", dest="quantize", type=str2bool, default=None, help='Quantize the model')
#parser.add_argument('--model_surgery', type=str, default=None, choices=[None, 'pact2'], help='whether to transform the model after defining')
parser.add_argument('--pretrained', type=str, default=None, help='pretrained model')
parser.add_argument('--resume', type=str, default=None, help='resume an unfinished training from this model')
parser.add_argument('--log_file', type=str, default=None, help='log_file')
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
from edgeai_tensorvision.xengine import test_pixel2pixel_onnx

# Create the parse and set default arguments
args = test_pixel2pixel_onnx.get_config()

#Modify arguments
args.model_name = "deeplabv3plus_edgeailite_mobilenetv2_relu" 

args.dataset_name = 'cityscapes_segmentation_measure' #'cityscapes_segmentation_infer' #'tiad_segmentation' #'tiad_segmentation_infer'   #

#args.save_path = './data/checkpoints/edgeailite'
args.data_path = './data/datasets/cityscapes/data'  #'./data/tiad/data/demoVideo/sequence0021'  #'./data/tiad/data/demoVideo/sequence0025'   #'./data/tiad/data/demoVideo/sequence0001_2017'
args.pretrained = './data/modelzoo/semantic_segmentation/cityscapes/deeplabv3plus_edgeailite-mobilenetv2/cityscapes_segmentation_deeplabv3plus_edgeailite-mobilenetv2_2019-06-26-08-59-32.onnx'

# For 19 class inference
palette19="[[128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],[0,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],[0,80,100],[0,0,230],[119,11,32]]"

# For 5 class inference
classes = ['road', 'sky', 'pedestrian', 'vehicle', 'background']
# palette19 = "[[128,64,128],[0,130,180],[220,20,60],[102,102,156],[190,153,153]]"
# palette19 = "[[128,64,128],[190,153,153],[220,20,60],[102,102,156],[190,153,153]]"

args.model_config.input_channels = (3,)

args.model_config.output_type = ['segmentation']
args.model_config.output_channels = None
args.losses = [['segmentation_loss']]
args.metrics = [['segmentation_metrics']]

args.num_images = 10000
args.blend = True #True
args.car_mask = False  # False   #True
args.label = True     # False   #True
args.palette = palette19
args.start_img_index = 0
args.end_img_index = 0
args.create_video = True  #True  #False

args.epoch_size = 0                     #0 #0.5
args.iter_size = 1                      #2

args.batch_size = 12 #80                  #16 #32 #64
args.img_resize = (384, 768)         #(256,512) #(512,512) # #(1024, 2048) #(512,1024)  #(720, 1280)
args.output_size = (1024, 2048)
#args.rand_scale = (1.0, 2.0)            #(1.0,2.0) #(1.0,1.5) #(1.0,1.25)


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
test_pixel2pixel_onnx.main(args)

