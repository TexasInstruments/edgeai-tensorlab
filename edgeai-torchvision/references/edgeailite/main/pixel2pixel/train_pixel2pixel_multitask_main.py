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
parser.add_argument('--lr', type=float, default=None, help='Base learning rate')
parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay for optimization')
parser.add_argument('--lr_clips', type=float, default=None, help='Learning rate for clips in PAct2')
parser.add_argument('--lr_calib', type=float, default=None, help='Learning rate for calibration')
parser.add_argument('--model_name', type=str, default=None, help='model name')
parser.add_argument('--dataset_name', type=str, default=None, help='dataset name')
parser.add_argument('--image_folders', type=str, default=('leftImg8bit_flow_confidence_768x384', 'leftImg8bit'), nargs='*', help='image_folders')
parser.add_argument('--data_path', type=str, default=None, help='data path')
parser.add_argument('--epoch_size', type=float, default=None, help='epoch size. using a fraction will reduce the data used for one epoch')
parser.add_argument('--epochs', type=int, default=None, help='number of epochs')
parser.add_argument('--warmup_epochs', type=int, default=None, help='number of epochs for the learning rate to increase and reach base value')
parser.add_argument('--milestones', type=int, nargs='*', default=None, help='change lr at these milestones')
parser.add_argument('--img_resize', type=int, nargs=2, default=None, help='img_resize size. for training this will be modified according to rand_scale')
parser.add_argument('--rand_scale', type=float, nargs=2, default=None, help='random scale factors for training')
parser.add_argument('--rand_crop', type=int, nargs=2, default=None, help='random crop for training')
parser.add_argument('--output_size', type=int, nargs=2, default=None, help='output size of the evaluation - prediction/groundtruth. this is not used while training as it blows up memory requirement')
parser.add_argument('--pretrained', type=str, default=None, help='pretrained model')
parser.add_argument('--resume', type=str, default=None, help='resume an unfinished training from this model')
parser.add_argument('--phase', type=str, default=None, help='training/calibration/validation')
parser.add_argument('--evaluate_start', type=str2bool, default=None, help='Whether to run validation before the training')
#
parser.add_argument("--quantization", "--quantize", dest="quantize", type=str2bool, default=None, help='Quantize the model')
parser.add_argument('--histogram_range', type=str2bool, default=None, help='run only evaluation and no training')
parser.add_argument('--per_channel_q', type=str2bool, default=None, help='run only evaluation and no training')
parser.add_argument('--bias_calibration', type=str2bool, default=None, help='run only evaluation and no training')
parser.add_argument('--bitwidth_weights', type=int, default=None, help='bitwidth for weight quantization')
parser.add_argument('--bitwidth_activations', type=int, default=None, help='bitwidth for activation quantization')
parser.add_argument('--multi_task_type', type=str, default=None, help='multi_task_type')
#
parser.add_argument('--freeze_bn', type=str2bool, default=None, help='freeze the bn stats or not')
cmds = parser.parse_args()

################################
# taken care first, since this has to be done before importing pytorch
if 'gpus' in vars(cmds):
    value = getattr(cmds, 'gpus')
    if (value is not None) and ("CUDA_VISIBLE_DEVICES" not in os.environ):
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(v) for v in value])
    #
#

################################
# to avoid hangs in data loader with multi threads
# this was observed after using cv2 image processing functions
# https://github.com/pytorch/pytorch/issues/1355
cv2.setNumThreads(0)


################################
#import of torch should be after CUDA_VISIBLE_DEVICES for it to take effect
import torch
from edgeai_xvision.xengine import train_pixel2pixel

#Create the parse and set default arguments
args = train_pixel2pixel.get_config()


#Modify arguments

args.model_name = 'deeplabv3plus_mobilenetv2_ericsun_edgeailite' 

args.dataset_name =  'cityscapes_depth_semantic_motion_multi_input' #'cityscapes_flow_depth_segmentation_image_pair' #cityscapes_segmentation #'cityscapes_segmentation_dual'

#args.save_path = './data/checkpoints/edgeailite'

args.data_path = './data/datasets/cityscapes/data'  #./data/pascal-voc/VOCdevkit/VOC2012

args.pretrained = '../edgeai-modelforest/models/vision/segmentation/cityscapes/edgeai-jai/deeplabv3plus_edgeailite_mobilenetv2_ericsun_768x384_20190411_checkpoint.pth'
                            #'./data/checkpoints/edgeailite/cityscapes_depth_semantic_five_class_motion_image_dof_conf/0p9_release/2019-06-27-13-50-10_cityscapes_depth_semantic_five_class_motion_image_dof_conf_deeplabv3plus_edgeailite_mobilenetv2_ericsun_mi4_resize768x384_traincrop768x384/model_best.pth.tar'
                            #'./data/modelzoo/pretrained/pytorch/cityscapes_segmentation/v0.9-2018-12-07-19:38:26_cityscapes_segmentation_deeplabv3plus_edgeailite_mobilenetv2_relu_resize768x384_traincrop768x384_(68.9%)/model_best.pth.tar'
                            #'./data/checkpoints/edgeailite/store/saved/cityscapes_segmentation/v0.7-2018-10-25-13:07:38_cityscapes_segmentation_deeplabv3plus_edgeailite_mobilenetv2_relu_resize1024x512_traincrop512x512_(71.5%)/model_best.pth.tar'
                            #'./data/modelzoo/pretrained/pytorch/imagenet_classification/pytorch_jacinto_ai.xvision/resnet50-19c8e357.pth'
                            #'./data/modelzoo/pretrained/pytorch/imagenet_classification/ericsun99/MobileNet-V2-Pytorch/mobilenetv2_Top1_71.806_Top2_90.410.pth.tar'

# args.resume = '/user/a0132471/Files/pytorch/pytorch-jacinto-models/checkpoints/cityscapes_depth_semantic_five_class_motion_image_dof_conf/2019-08-13-13-49-29_cityscapes_depth_semantic_five_class_motion_image_dof_conf_deeplabv3plus_edgeailite_mobilenetv2_ericsun_mi4_resize768x384_traincrop768x384/checkpoint.pth.tar'

args.model_config.input_channels = (3,3)
args.model_config.output_type = ['depth', 'segmentation', 'segmentation']
args.model_config.output_channels = None             #this can be found out from the dataset
args.losses = [['supervised_loss', 'scale_loss'], ['segmentation_loss'], ['segmentation_loss']]
args.metrics = [['supervised_relative_error_rng3to80'], ['segmentation_metrics'], ['segmentation_metrics']]
args.is_flow = [[True,False],[False,False,False]]


# TODO: this is not clean - fix it later
args.model_config.multi_task_type = cmds.multi_task_type #None, 'adaptive', 'learned', 'uncertainty'
args.model_config.multi_task = True if cmds.multi_task_type is not None else False
args.model_config.multi_task_factors = (1.0, 1.0, 1.0) #None #[1.291, 6.769, 6.852] (0.169, 1.279, 1.553)
args.multi_decoder = True



args.optimizer = 'adam'                    #'sgd' #'adam'
args.epochs = 250                       #200
args.epoch_size = 0                     #0 #0.5
args.scheduler = 'step'                 #'poly' #'step'
args.multistep_gamma = 0.5              #only for step scheduler
args.milestones = (100, 200)            #only for step scheduler
args.polystep_power = 0.9               #only for poly scheduler
args.iter_size = 1                      #2
args.evaluate_start = False


args.lr = 1e-4                          #1e-4 #0.01 #7e-3 #1e-4 #2e-4
args.batch_size = 16                    #12 #16 #32 #64
args.weight_decay = 1e-4                #4e-5 #1e-5

args.img_resize = (384, 768)            #(512, 1024) #(1024, 2048)
args.rand_scale = (1.0, 2.0)            #(1.0,2.0)
args.rand_crop = (384, 768)             #(512,512) #(512,1024)
args.output_size = (1024, 2048)          #for unflow loss only, output_size must match img_size

args.transform_rotation = 5             #0  #rotation degrees

args.model_config.aspp_dil = (2, 4, 6)
#args.model_config.use_aspp = True

# TODO: this is not clean - fix it later
args.dataset_config.image_folders = cmds.image_folders

args.save_onnx = False
#args.phase = 'validation'
#args.quantize = True

args.model_config.normalize_gradients = True

args.pivot_task_idx = 2

################################
for key in vars(cmds):
    if (key == 'gpus')| (key == 'image_folders') | (key == 'multi_task_type'):
        pass # already taken care above, since this has to be done before importing pytorch
    elif hasattr(args, key):
        value = getattr(cmds, key)
        if value != 'None' and value is not None:
            setattr(args, key, value)
    else:
        assert False, f'invalid argument {key}'
#

################################
# Run the given phase
train_pixel2pixel.main(args)

################################
# In addition run a quantization aware training, starting from the trained model
if 'training' in args.phase and (not args.quantize):
    save_path = train_pixel2pixel.get_save_path(args)
    args.pretrained = os.path.join(save_path, 'model_best.pth.tar') if (args.epochs>0) else args.pretrained
    args.phase = 'training_quantize'
    args.quantize = True
    args.lr = 1e-5
    args.epochs = 10
    train_pixel2pixel.main(args)
#

################################
# In addition run a separate validation
if 'training' in args.phase or 'calibration' in args.phase:
    save_path = train_pixel2pixel.get_save_path(args)
    args.pretrained = os.path.join(save_path, 'model_best.pth.tar')
    if 'training' in args.phase:
        # DataParallel isn't enabled for QuantCalibrateModule and QuantTestModule.
        # If the previous phase was training, then it is likely that the batch_size was high and won't fit in a single gpu - reduce it.
        num_gpus = len(str(os.environ["CUDA_VISIBLE_DEVICES"]).split(',')) if ("CUDA_VISIBLE_DEVICES" in os.environ) else None
        args.batch_size = max(args.batch_size//num_gpus, 1) if (num_gpus is not None) else args.batch_size
    #
    args.phase = 'validation'
    args.quantize = True
    train_pixel2pixel.main(args)
#
