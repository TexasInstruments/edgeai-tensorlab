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
parser.add_argument('--image_mean', type=float, nargs='*', default=None, help='image_mean')
parser.add_argument('--image_scale', type=float, nargs='*', default=None, help='image_scale')
parser.add_argument('--input_channel_reverse', type=str2bool, default=None, help='input_channel_reverse, for example rgb to bgr')
parser.add_argument('--save_path', type=str, default=None, help='checkpoint save folder')
parser.add_argument('--gpus', type=int, nargs='*', default=None, help='Base learning rate')
parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
parser.add_argument('--strides', type=int, nargs='*', default=None, help='strides in the model')
parser.add_argument('--lr', type=float, default=None, help='Base learning rate')
parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay for optimization')
parser.add_argument('--lr_clips', type=float, default=None, help='Learning rate for clips in PAct2')
parser.add_argument('--lr_calib', type=float, default=None, help='Learning rate for calibration')
parser.add_argument('--model_name', type=str, default=None, help='model name')
parser.add_argument('--model', default=None, help='model')
parser.add_argument('--dataset_name', type=str, default=None, help='dataset name')
parser.add_argument('--data_path', type=str, default=None, help='data path')
parser.add_argument('--optimizer', type=str, default=None, help='optimizer/solver type: sgd, adam')
parser.add_argument('--scheduler', type=str, default=None, help='lr scheduler: step, cosine')
parser.add_argument('--epochs', type=int, default=None, help='number of epochs')
parser.add_argument('--warmup_epochs', type=int, default=None, help='number of epochs for the learning rate to increase and reach base value')
parser.add_argument('--milestones', type=int, nargs='*', default=None, help='change lr at these milestones')

parser.add_argument('--img_resize', type=int, default=None, help='images will be first resized to this size during training and validation')
parser.add_argument('--rand_scale', type=float, nargs=2, default=None, help='during training (only) fraction of the image to crop (this will then be resized to img_crop)')
parser.add_argument('--img_crop', type=int, default=None, help='the cropped portion (validation), cropped pertion will be resized to this size (training)')
parser.add_argument('--auto_augument', type=str, default=None, help='auto augument: learning augumentation strategies from data - '
                                                                    'https://arxiv.org/abs/1805.09501'
                                                                    'currently supported options are: imagenet, cifar10, svhn'
                                                                    'note: more epochs of training may be required if this augumentation is used')
parser.add_argument('--random_erasing', type=float, default=None, help='random erasing data augumentation - https://arxiv.org/abs/1708.04896'
                                                                       'suggested value - 0.2')

parser.add_argument('--pretrained', type=str, default=None, help='pretrained model')
parser.add_argument('--resume', type=str, default=None, help='resume an unfinished training from this model')
parser.add_argument('--phase', type=str, default=None, help='training/calibration/validation')
parser.add_argument('--evaluate_start', type=str2bool, default=None, help='Whether to run validation before the training')
parser.add_argument('--workers', type=int, default=None, help='number of workers for dataloading')
parser.add_argument('--save_onnx', type=str2bool, default=None, help='whether to export/save onnx graph or not')
#
parser.add_argument("--quantization", "--quantize", dest="quantize", type=str2bool, default=None, help='Quantize the model')
parser.add_argument('--histogram_range', type=str2bool, default=None, help='run only evaluation and no training')
parser.add_argument('--per_channel_q', type=str2bool, default=None, help='run only evaluation and no training')
parser.add_argument('--bias_calibration', type=str2bool, default=None, help='run only evaluation and no training')
parser.add_argument('--bitwidth_weights', type=int, default=None, help='bitwidth for weight quantization')
parser.add_argument('--bitwidth_activations', type=int, default=None, help='bitwidth for activation quantization')
parser.add_argument('--constrain_bias', type=int, default=None, help='constrain_bias')
#
parser.add_argument('--freeze_bn', type=str2bool, default=None, help='freeze the bn stats or not')
#
parser.add_argument('--shuffle', type=str2bool, default=None, help='whether to shuffle the training set or not')
parser.add_argument('--shuffle_val', type=str2bool, default=None, help='whether to shuffle the validation set or not')
parser.add_argument('--epoch_size', type=float, default=None, help='epoch size. options are: 0, fraction or number. '
                                                                   '0 will use the full epoch. '
                                                                   'using a number will cause the epoch to have that many images. '
                                                                   'using a fraction will reduce the number of images used for one epoch. ')
parser.add_argument('--epoch_size_val', type=float, default=None, help='epoch size for validation. options are: 0, fraction or number. '
                                                                   '0 will use the full epoch. '
                                                                   'using a number will cause the epoch to have that many images. '
                                                                   'using a fraction will reduce the number of images used for one epoch. ')
parser.add_argument('--parallel_model', type=str2bool, default=None, help='whether to use DataParallel for models')
parser.add_argument('--enable_fp16', type=str2bool, default=None, help='fp16/half precision mode to speedup training')
#
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
from edgeai_tensorvision.xengine import train_classification

#Create the parse and set default arguments
args = train_classification.get_config()

################################
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

################################
#Set arguments
args.model_name = 'mobilenetv2_tv_x1'        # 'resnet50_x1', 'mobilenetv2_tv_x1', 'mobilenetv2_ericsun_x1', 'mobilenetv2_shicai_x1'

args.dataset_name = 'image_folder_classification' # 'image_folder_classification', 'imagenet_classification', 'cifar10_classification', 'cifar100_classification'

args.data_path = f'./data/datasets/{args.dataset_name}'

#args.pretrained = 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
                    #'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
                    #'./data/modelzoo/pretrained/pytorch/imagenet_classification/torchvision/resnet50-19c8e357.pth'
                    #'./data/modelzoo/pretrained/pytorch/imagenet_classification/ericsun99/MobileNet-V2-Pytorch/mobilenetv2_Top1_71.806_Top2_90.410.pth.tar'

#args.resume = './data/checkpoints/edgeailite/imagenet_classification/2019-01-24-09:34:45_imagenet_classification_shufflemobilenetv2_resize256_crop224_(epochs29)/checkpoint.pth.tar'


# can specify parameter overrides here as well, if you don't want to paas it via commandline arguments
#args.save_path = './data/checkpoints/edgeailite'
#args.model_config.input_channels = 3
#args.model_config.output_type = 'classification'
#args.model_config.output_channels = None
#args.model_config.strides = None #(2,2,2,2,2)
#args.img_resize = 256
#args.img_crop = 224
#args.optimizer = 'sgd'                  #'sgd' #'adam'
#args.epochs = 150                       #150 #120
#args.start_epoch = 0                    #0
#args.epoch_size = 0                     #0 #0.5
#args.scheduler = 'cosine'               #'poly' #'step' #'exponential' #'cosine'
#args.multistep_gamma = 0.1              #0.1 #0.94 #0.98 #for step and exponential schedulers
#args.milestones = [30, 60, 90]          #only for step scheduler
#args.polystep_power = 1.0               #only for poly scheduler
#args.step_size = 1                      #only for exp scheduler
#args.iter_size = 1                      #2
#args.warmup_epochs = 5                  #5
#args.lr = 0.1                           #0.2 #0.1 #0.045 #0.01
#args.batch_size = 512 #256 #128 #512 #1024
#args.weight_decay = 4e-5                #4e-5 #1e-4
#args.rand_scale = (0.2, 1.0)            #(0.08, 1.0)
#args.print_model = True
#args.save_onnx = False
#args.run_soon = True #default(True) #Set to false if only cfs files/onnx  models needed but no training is required
#args.phase = 'validation'
#args.evaluate_start = False
#args.start_epoch = 100
#args.quantize = True
#args.phase = 'training'

# defining date from outside can help to write multiple pahses into the same folder
args.date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

################################
for key in vars(cmds):
    if key == 'gpus': # already taken care above, since this has to be done before importing pytorch
        pass
    elif key in ('strides', 'enable_fp16'): # these are in model_config
        value = getattr(cmds, key)
        if value != 'None' and value is not None:
            setattr(args.model_config, key, value)
        #
    elif hasattr(args, key):
        value = getattr(cmds, key)
        if value != 'None' and value is not None:
            setattr(args, key, value)
    else:
        assert False, f'invalid argument {key}'
#

################################
# num_classes depends on the dataset chosen
args.model_config.num_classes = (100 if 'cifar100' in args.dataset_name else (10 if 'cifar10' in args.dataset_name else 1000))

################################
# Run the training
train_classification.main(args)

################################
# if the previous phase was training, run a quantization aware training, starting from the trained model
if 'training' in args.phase and (not args.quantize):
    if args.epochs > 0:
        save_path = train_classification.get_save_path(args)
        if isinstance(args.model, str) and args.model.endswith('.onnx'):
            args.model = os.path.join(save_path, 'model_best.onnx')
        #
        args.pretrained = os.path.join(save_path, 'model_best.pth')
    #
    args.phase = 'training_quantize'
    args.quantize = True
    args.lr = 1e-5
    args.epochs = 10
    train_classification.main(args)
#

################################
# In addition, run a separate validation starting from the calibrated model - to estimate the quantized accuracy accurately
if 'training' in args.phase or 'calibration' in args.phase:
    save_path = train_classification.get_save_path(args)
    if isinstance(args.model, str) and args.model.endswith('.onnx'):
        args.model = os.path.join(save_path, 'model_best.onnx')
    #
    args.pretrained = os.path.join(save_path, 'model_best.pth')
    if 'training' in args.phase:
        # DataParallel isn't enabled for QuantCalibrateModule and QuantTestModule.
        # If the previous phase was training, then it is likely that the batch_size was high and won't fit in a single gpu - reduce it.
        num_gpus = len(str(os.environ["CUDA_VISIBLE_DEVICES"]).split(',')) if ("CUDA_VISIBLE_DEVICES" in os.environ) else None
        args.batch_size = max(args.batch_size//num_gpus, 1) if (num_gpus is not None) else args.batch_size
    #
    args.phase = 'validation'
    args.quantize = True
    train_classification.main(args)
#


