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
from edgeai_tensorvision.xengine import train_pixel2pixel


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_mean', type=float, nargs='*', default=None, help='image_mean')
    parser.add_argument('--image_scale', type=float, nargs='*', default=None, help='image_scale')
    parser.add_argument('--input_channel_reverse', type=str2bool, default=None,
                        help='input_channel_reverse, for example rgb to bgr')
    parser.add_argument('--save_path', type=str, default=None, help='checkpoint save folder')
    parser.add_argument('--gpus', type=str, nargs='*', default=None, help='Base learning rate')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
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
    parser.add_argument('--warmup_epochs', type=int, default=None,
                        help='number of epochs for the learning rate to increase and reach base value')
    parser.add_argument('--milestones', type=int, nargs='*', default=None, help='change lr at these milestones')
    parser.add_argument('--multistep_gamma', type=float, default=None, help='multistep_gamma fro step lr adjustment')
    parser.add_argument('--img_resize', type=int, nargs=2, default=None,
                        help='img_resize size. for training this will be modified according to rand_scale')
    parser.add_argument('--rand_scale', type=float, nargs=2, default=None, help='random scale factors for training')
    parser.add_argument('--rand_crop', type=int, nargs=2, default=None, help='random crop for training')
    parser.add_argument('--output_size', type=int, nargs=2, default=None,
                        help='output size of the evaluation - prediction/groundtruth. this is not used while training as it blows up memory requirement')
    parser.add_argument('--pretrained', type=str, default=None, help='pretrained model')
    parser.add_argument('--resume', type=str, default=None, help='resume an unfinished training from this model')
    parser.add_argument('--phase', type=str, default=None, help='training/calibration/validation')
    parser.add_argument('--evaluate_start', type=str2bool, default=None,
                        help='Whether to run validation before the training')
    parser.add_argument('--workers', type=int, default=None, help='number of workers for dataloading')
    parser.add_argument('--save_onnx', type=str2bool, default=None, help='Whether to export onnx model or not')
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
    parser.add_argument('--epoch_size_val', type=float, default=None,
                        help='epoch size for validation. options are: 0, fraction or number. '
                             '0 will use the full epoch. '
                             'using a number will cause the epoch to have that many images. '
                             'using a fraction will reduce the number of images used for one epoch. ')
    parser.add_argument('--interpolation', type=int, default=None,
                        help='interpolation mode to be used from one of cv2.INTER_ modes')
    parser.add_argument('--parallel_model', type=str2bool, default=None, help='whether to use DataParallel for models')
    parser.add_argument('--img_border_crop', type=int, nargs=4, default=None,
                        help='image border crop rectangle. can be relative or absolute')
    parser.add_argument('--enable_fp16', type=str2bool, default=None, help='fp16/half precision mode to speedup training')
    parser.add_argument('--annotation_prefix', type=str, default=None, help='Annotation Prefix Name')
    parser.add_argument('--output_dir', type=str, default=None, help='Output Directory')
    parser.add_argument('--device', type=str, default=None, help='Device Name')
    parser.add_argument('--distributed', type=str2bool, default=False, help='Distributed training check')
    #
    return parser


################################
def main(arguemnts):

    # Run the given phase
    train_pixel2pixel.main(arguemnts)

    ################################
    # if the previous phase was training, run a quantization aware training, starting from the trained model
    if 'training' in arguemnts.phase and (arguemnts.quantize): # Removed not in arguemnts.quantize
        if arguemnts.epochs > 0:
            if arguemnts.save_path is None:
                save_path = train_pixel2pixel.get_save_path(arguemnts)
            else:
                save_path = arguemnts.save_path
            if isinstance(arguemnts.model, str) and arguemnts.model.endswith('.onnx'):
                # arguemnts.model = os.path.join(save_path, 'model.onnx')
                arguemnts.model = os.path.join(save_path, 'model_best.onnx')
            #
            # arguemnts.pretrained = os.path.join(save_path, 'model.pth')
            arguemnts.pretrained = os.path.join(save_path, 'model_best.pth')
        #
        arguemnts.phase = 'training_quantize'
        arguemnts.quantize = False
        arguemnts.lr = 1e-5
        arguemnts.epochs = 10
        train_pixel2pixel.main(arguemnts)
    #

    ################################
    # In addition run a separate validation
    if 'validation' in arguemnts.phase: #'training' in arguemnts.phase or 'calibration' in arguemnts.phase:
        if arguemnts.save_path is None:
            save_path = train_pixel2pixel.get_save_path(arguemnts)
        else:
            save_path = arguemnts.save_path
        if isinstance(arguemnts.model, str) and arguemnts.model.endswith('.onnx'):
            # arguemnts.model = os.path.join(save_path, 'model.onnx')
            arguemnts.model = os.path.join(save_path, 'model_best.onnx')

        # arguemnts.pretrained = os.path.join(save_path, 'model.pth')
        arguemnts.pretrained = os.path.join(save_path, 'model_best.pth')

        if 'training' in arguemnts.phase:
            # DataParallel isn't enabled for QuantCalibrateModule and QuantTestModule.
            # If the previous phase was training, then it is likely that the batch_size was high and won't fit in a single gpu - reduce it.
            num_gpus = len(str(os.environ["CUDA_VISIBLE_DEVICES"]).split(',')) if ("CUDA_VISIBLE_DEVICES" in os.environ) else None
            arguemnts.batch_size = max(arguemnts.batch_size // num_gpus, 1) if (num_gpus is not None) else arguemnts.batch_size

        arguemnts.phase = 'validation'
        arguemnts.quantize = False
        train_pixel2pixel.main(arguemnts)


def run(arg):
    ################################
    # taken care first, since this has to be done before importing pytorch
    if 'gpus' in vars(arg):
        value = getattr(arg, 'gpus')
        if (value is not None) and ("CUDA_VISIBLE_DEVICES" not in os.environ):
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(v) for v in str(value[0]).split())
        #
    #

    ################################
    # to avoid hangs in data loader with multi threads
    # this was observed after using cv2 image processing functions
    # https://github.com/pytorch/pytorch/issues/1355
    cv2.setNumThreads(0)

    ################################
    # import of torch should be after CUDA_VISIBLE_DEVICES for it to take effect

    # Create the parser and set default arguments
    arguments = train_pixel2pixel.get_config()

    ################################
    # Modify arguments
    arguments.model_name = 'deeplabv3plus_mobilenetv2_tv_edgeailite'  # 'deeplabv3plus_mobilenetv2_tv_edgeailite' #'fpn_aspp_mobilenetv2_tv_edgeailite' #'fpn_aspp_resnet50_edgeailite'
    arguments.dataset_name = 'cityscapes_segmentation'  # 'cityscapes_segmentation' #'voc_segmentation'

    arguments.data_path = './data/datasets/cityscapes/data'  # './data/datasets/cityscapes/data' #'./data/datasets/voc'

    # args.save_path = './data/checkpoints/edgeailite'

    arguments.pretrained = 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
    # 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
    # './data/modelzoo/pretrained/pytorch/imagenet_classification/ericsun99/MobileNet-V2-Pytorch/mobilenetv2_Top1_71.806_Top2_90.410.pth.tar'
    # 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

    # args.resume = './data/checkpoints/edgeailite/cityscapes_segmentation/2019-04-11-05-35-55_cityscapes_segmentation_deeplabv3plus_edgeailite_mobilenetv2_relu_resize768x384_traincrop768x384/checkpoint.pth.tar'

    arguments.model_config.input_channels = (3,)
    arguments.model_config.output_type = ['segmentation']
    arguments.model_config.output_channels = None
    arguments.model_config.output_range = None
    arguments.model_config.num_decoders = None  # 0, 1, None

    arguments.losses = [['segmentation_loss']]
    arguments.metrics = [['segmentation_metrics']]

    arguments.optimizer = 'adam'  # 'sgd' #'adam'
    arguments.epochs = 250  # 200
    arguments.epoch_size = 0  # 0 #0.5
    arguments.epoch_size_val = 0  # 0 #0.5
    arguments.scheduler = 'step'  # 'poly' #'step'
    arguments.multistep_gamma = 0.25  # 0.5 #only for step scheduler
    arguments.milestones = (100, 200)  # only for step scheduler
    arguments.polystep_power = 0.9  # only for poly scheduler
    arguments.iter_size = 1  # 2

    arguments.lr = 4e-4  # 1e-4 #0.01 #7e-3 #1e-4 #2e-4
    arguments.batch_size = 12  # 12 #16 #32 #64
    arguments.weight_decay = 1e-4  # 1e-4  #4e-5 #1e-5

    arguments.img_resize = (512,512)  # (384, 768) (512, 1024) #(1024, 2048)
    arguments.output_size = (512,512)  # target output size for evaluation

    arguments.transform_rotation = 5  # rotation degrees

    # args.image_mean = [123.675, 116.28, 103.53]
    # args.image_scale = [0.017125, 0.017507, 0.017429]

    # args.parallel_model=False
    # args.print_model = True
    # args.save_onnx = False
    # args.run_soon = False
    # args.evaluate_start = False
    arguments.print_freq = 10

    # args.phase = 'validation' #'training'
    # args.quantize = True
    # args.per_channel_q = True


    # defining date from outside can help to write multiple phases into the same folder
    arguments.date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    ################################
    # set other args
    for key in vars(arg):
        if key == 'gpus':
            value = getattr(arg, key)
            if value != 'None' and value is not None:
                setattr(arguments, key, value)
            #
        elif key in ('strides', 'enable_fp16'):  # these are in model_config
            value = getattr(arg, key)
            if value != 'None' and value is not None:
                setattr(arguments.model_config, key, value)
            #
        elif hasattr(arguments, key):
            value = getattr(arg, key)
            if value != 'None' and value is not None:
                setattr(arguments, key, value)
        else:
            assert False, f'invalid argument {key}'

    main(arguments)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    run(args)
