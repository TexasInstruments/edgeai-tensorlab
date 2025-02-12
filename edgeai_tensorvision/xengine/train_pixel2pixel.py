# Copyright (c) 2018-2021, Texas Instruments
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

import os
import shutil
import time
import math
import copy

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.onnx
import onnx

import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import cv2
from colorama import Fore
import progiter
from packaging import version
import warnings

import edgeai_torchmodelopt
from edgeai_torchmodelopt import xnn
from edgeai_tensorvision import xvision
from edgeai_tensorvision.xvision.transforms import image_transforms
from edgeai_tensorvision.xvision import losses as pixel2pixel_losses
from .infer_pixel2pixel import compute_accuracy

##################################################
warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)

##################################################
def get_config():
    args = xnn.utils.ConfigNode()

    args.dataset_config = xnn.utils.ConfigNode()
    args.dataset_config.split_name = 'val'
    args.dataset_config.max_depth_bfr_scaling = 80
    args.dataset_config.depth_scale = 1
    args.dataset_config.train_depth_log = 1
    args.use_semseg_for_depth = False

    # model config
    args.model_config = xnn.utils.ConfigNode()
    args.model_config.output_type = ['segmentation']   # the network is used to predict flow or depth or sceneflow
    args.model_config.output_channels = None            # number of output channels
    args.model_config.prediction_channels = None        # intermediate number of channels before final output_channels
    args.model_config.input_channels = None             # number of input channels
    args.model_config.final_upsample = True             # use final upsample to input resolution or not
    args.model_config.output_range = None               # max range of output
    args.model_config.num_decoders = None               # number of decoders to use. [options: 0, 1, None]
    args.model_config.freeze_encoder = False            # do not update encoder weights
    args.model_config.freeze_decoder = False            # do not update decoder weights
    args.model_config.multi_task_type = 'learned'       # find out loss multiplier by learning, choices=[None, 'learned', 'uncertainty', 'grad_norm', 'dwa_grad_norm']
    args.model_config.target_input_ratio = 1            # Keep target size same as input size
    args.model_config.input_nv12 = False                # convert input to nv12 format
    args.model_config.enable_fp16 = False               # faster training if the GPU supports fp16

    args.model = None                                   # the model itself can be given from ouside
    args.model_name = 'deeplabv2lite_mobilenetv2'       # model architecture, overwritten if pretrained is specified
    args.dataset_name = 'cityscapes_segmentation'       # dataset type
    args.transforms = None                              # the transforms itself can be given from outside
    args.input_channel_reverse = False                  # reverse input channels, for example RGB to BGR
    args.annotation_prefix = "instances"                # Annotations prefix name

    args.data_path = './data/cityscapes'                # 'path to dataset'
    args.save_path = None                               # checkpoints save path
    args.phase = 'training'                             # training/calibration/validation
    args.date = None                                    # date to add to save path. if this is None, current date will be added.
    args.output_dir = None

    args.logger = None                                  # logger stream to output into
    args.show_gpu_usage = False                         # Shows gpu usage at the begining of each training epoch
    args.device = None
    args.distributed = None

    args.split_file = None                              # train_val split file
    args.split_files = None                             # split list files. eg: train.txt val.txt
    args.split_value = None                             # test_val split proportion (between 0 (only test) and 1 (only train))

    args.optimizer = 'adam'                                # optimizer algorithms, choices=['adam','sgd']
    args.scheduler = 'step'                             # scheduler algorithms, choices=['step','poly', 'cosine']
    args.workers = 8                                    # number of data loading workers

    args.epochs = 250                                   # number of total epochs to run
    args.start_epoch = 0                                # manual epoch number (useful on restarts)

    args.epoch_size = 0                                 # manual epoch size (will match dataset size if not specified)
    args.epoch_size_val = 0                             # manual epoch size (will match dataset size if not specified)
    args.batch_size = 12                                # mini_batch size
    args.total_batch_size = None                        # accumulated batch size. total_batch_size = batch_size*iter_size
    args.iter_size = 1                                  # iteration size. total_batch_size = batch_size*iter_size

    args.lr = 1e-4                                      # initial learning rate
    args.lr_clips = None                                # use args.lr itself if it is None
    args.lr_calib = 0.05                                # lr for bias calibration
    args.warmup_epochs = 5                              # number of epochs to warmup
    args.warmup_factor = 1e-3                           # max lr allowed for the first epoch during warmup (as a factor of initial lr)

    args.momentum = 0.9                                 # momentum for sgd, alpha parameter for adam
    args.beta = 0.999                                   # beta parameter for adam
    args.weight_decay = 1e-4                            # weight decay
    args.bias_decay = None                              # bias decay

    args.sparse = True                                  # avoid invalid/ignored target pixels from loss computation, use NEAREST for interpolation

    args.tensorboard_num_imgs = 5                       # number of imgs to display in tensorboard
    args.pretrained = None                              # path to pre_trained model
    args.resume = None                                  # path to latest checkpoint (default: none)
    args.no_date = False                                # don\'t append date timestamp to folder
    args.print_freq = 100                               # print frequency (default: 100)

    args.milestones = (100, 200)                        # epochs at which learning rate is divided by 2

    args.losses = ['segmentation_loss']                 # loss functions to mchoices=['step','poly', 'cosine'],loss multiplication factor')
    args.metrics = ['segmentation_metrics']  # metric/measurement/error functions for train/validation
    args.multi_task_factors = None                      # loss mult factors
    args.class_weights = None                           # class weights

    args.loss_mult_factors = None                       # fixed loss mult factors - per loss - not: this is different from multi_task_factors (which is per task)

    args.multistep_gamma = 0.5                          # steps for step scheduler
    args.polystep_power = 1.0                           # power for polynomial scheduler

    args.rand_seed = 1                                  # random seed
    args.img_border_crop = None                         # image border crop rectangle. can be relative or absolute
    args.target_mask = None                              # mask rectangle. can be relative or absolute. last value is the mask value

    args.rand_resize = None                             # random image size to be resized to during training
    args.rand_output_size = None                        # output size to be resized to during training
    args.rand_scale = (1.0, 2.0)                        # random scale range for training
    args.rand_crop = None                               # image size to be cropped to

    args.img_resize = None                              # image size to be resized to during evaluation
    args.output_size = None                             # target output size to be resized to

    args.count_flops = True                             # count flops and report

    args.shuffle = True                                 # shuffle or not
    args.shuffle_val = True                             # shuffle val dataset or not

    args.transform_rotation = 0.                        # apply rotation augumentation. value is rotation in degrees. 0 indicates no rotation
    args.is_flow = None                                 # whether entries in images and targets lists are optical flow or not

    args.upsample_mode = 'bilinear'                     # upsample mode to use, choices=['nearest','bilinear']

    args.image_prenorm = True                           # whether normalization is done before all other the transforms
    args.image_mean = (128.0,)                          # image mean for input image normalization
    args.image_scale = (1.0 / (0.25 * 256),)            # image scaling/mult for input iamge normalization

    args.max_depth = 80                                 # maximum depth to be used for visualization

    args.pivot_task_idx = 0                             # task id to select best model

    args.parallel_model = True                          # Usedata parallel for model
    args.parallel_criterion = True                      # Usedata parallel for loss and metric

    args.evaluate_start = True                          # evaluate right at the begining of training or not
    args.save_onnx = True                               # apply quantized inference or not
    args.print_model = False                            # print the model to text
    args.run_soon = True                                # To start training after generating configs/models

    args.quantize = False                               # apply quantized inference or not
    #args.model_surgery = None                           # replace activations with PAct2 activation module. Helpful in quantized training.
    args.bitwidth_weights = 8                           # bitwidth for weights
    args.bitwidth_activations = 8                       # bitwidth for activations
    args.histogram_range = True                         # histogram range for calibration
    args.bias_calibration = True                        # apply bias correction during quantized inference calibration
    args.per_channel_q = False                          # apply separate quantizion factor for each channel in depthwise or not
    args.constrain_bias = None                          # constrain bias according to the constraints of convolution engine

    args.save_mod_files = False                         # saves modified files after last commit. Also  stores commit id.
    args.make_score_zero_mean = False                   # make score zero mean while learning
    args.no_q_for_dws_layer_idx = 0                     # no_q_for_dws_layer_idx

    args.viz_colormap = 'rainbow'                       # colormap for tensorboard: 'rainbow', 'plasma', 'magma', 'bone'

    args.freeze_bn = False                              # freeze the statistics of bn
    args.tensorboard_enable = True                      # en/disable of TB writing
    args.print_train_class_iou = False
    args.print_val_class_iou = False
    args.freeze_layers = None
    args.opset_version = 11                             # onnx opset_version
    args.prob_color_to_gray = (0.0,0.0)                 # this will be used for controlling color 2 gray augmentation

    args.interpolation = None                           # interpolation method to be used for resize. one of cv2.INTER_
    return args


# ################################################
# to avoid hangs in data loader with multi threads
# this was observed after using cv2 image processing functions
# https://github.com/pytorch/pytorch/issues/1355
cv2.setNumThreads(0)

# ################################################
def main(args):
    # ensure pytorch version is 1.2 or higher
    assert version.parse(torch.__version__) >= version.parse('1.1'), \
        'torch version must be 1.1 or higher, due to the change in scheduler.step() and optimiser.step() call order'

    assert (not hasattr(args, 'evaluate')), 'args.evaluate is deprecated. use args.phase=training or calibration or validation'
    assert is_valid_phase(args.phase), f'invalid phase {args.phase}'
    assert not hasattr(args, 'model_surgery'), 'the argument model_surgery is deprecated, it is not needed now - remove it'

    if (args.phase == 'validation' and args.bias_calibration):
        args.bias_calibration = False
        warnings.warn('switching off bias calibration in validation')
    #

    to_device = lambda src_object, non_blocking=False: src_object.cuda(non_blocking=non_blocking) if args.device in ('cuda', None) else src_object
    module_to_device = lambda src_object, non_blocking=False: src_object.cuda() if args.device in ('cuda', None) else src_object

    #################################################
    args.rand_resize = args.img_resize if args.rand_resize is None else args.rand_resize
    args.rand_crop = args.img_resize if args.rand_crop is None else args.rand_crop
    args.output_size = args.img_resize if args.output_size is None else args.output_size
    # resume has higher priority
    args.pretrained = None if (args.resume is not None) else args.pretrained

    # prob_color_to_gray will be used for controlling color 2 gray augmentation
    if 'tiad' in args.dataset_name and args.prob_color_to_gray == (0.0, 0.0):
        #override in case of 'tiad' if default values are used
        args.prob_color_to_gray = (0.5, 0.0)

    if args.save_path is None:
        save_path = get_save_path(args)
    else:
        save_path = args.save_path
    #
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.save_mod_files:
        #store all the files after the last commit.
        mod_files_path = save_path+'/mod_files'
        os.makedirs(mod_files_path)
        
        cmd = "git ls-files --modified | xargs -i cp {} {}".format("{}", mod_files_path)
        print("cmd:", cmd)    
        os.system(cmd)

        #stoe last commit id. 
        cmd = "git log -n 1  >> {}".format(mod_files_path + '/commit_id.txt')
        print("cmd:", cmd)    
        os.system(cmd)

    #################################################
    if args.logger is None:
        args.logger = xnn.utils.TeeLogger(filename=os.path.join(save_path,'run.log'))

    #################################################
    # global settings. rand seeds for repeatability
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    torch.cuda.manual_seed(args.rand_seed)

    ################################
    # args check and config
    if args.iter_size != 1 and args.total_batch_size is not None:
        warnings.warn("only one of --iter_size or --total_batch_size must be set")
    #
    if args.total_batch_size is not None:
        args.iter_size = args.total_batch_size//args.batch_size
    else:
        args.total_batch_size = args.batch_size*args.iter_size

    #################################################
    # set some global flags and initializations
    # keep it in args for now - although they don't belong here strictly
    # using pin_memory is seen to cause issues, especially when when lot of memory is used.
    args.use_pinned_memory = False
    args.n_iter = 0
    args.best_metric = -1
    cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(True)

    ################################
    # reset character color, in case it is different
    print('{}'.format(Fore.RESET))
    # print everything for log
    print('=> args: {}'.format(args))
    print('\n'.join("%s: %s" % item for item in sorted(vars(args).items())))

    print('=> will save everything to {}'.format(save_path))

    #################################################
    train_writer = SummaryWriter(os.path.join(save_path,'train')) if args.tensorboard_enable else None
    val_writer = SummaryWriter(os.path.join(save_path,'val')) if args.tensorboard_enable else None
    transforms = get_transforms(args) if args.transforms is None else args.transforms
    assert isinstance(transforms, (list,tuple)) and len(transforms) == 2, 'incorrect transforms were given'

    print("=> fetching images in '{}'".format(args.data_path))
    split_arg = args.split_file if args.split_file else (args.split_files if args.split_files else args.split_value)
    train_dataset, val_dataset = xvision.datasets.__dict__[args.dataset_name](args.dataset_config, args.data_path, split=split_arg, transforms=transforms, annotation_prefix=args.annotation_prefix)

    #################################################
    print('=> {} samples found, {} train samples and {} test samples '.format(len(train_dataset)+len(val_dataset),
        len(train_dataset), len(val_dataset)))
    train_sampler = get_dataset_sampler(train_dataset, args.epoch_size) if args.epoch_size != 0 else None
    shuffle_train = args.shuffle and (train_sampler is None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=args.use_pinned_memory, sampler=train_sampler, shuffle=shuffle_train)

    val_sampler = get_dataset_sampler(val_dataset, args.epoch_size_val) if args.epoch_size_val != 0 else None
    shuffle_val = args.shuffle_val and (val_sampler is None)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=args.use_pinned_memory, sampler=val_sampler, shuffle=shuffle_val)

    #################################################
    if (args.model_config.input_channels is None):
        args.model_config.input_channels = (3,)
        print("=> input channels is not given - setting to {}".format(args.model_config.input_channels))

    if (args.model_config.output_channels is None):
        if ('num_classes' in dir(train_dataset)):
            args.model_config.output_channels = train_dataset.num_classes()
        else:
            args.model_config.output_channels = (2 if args.model_config.output_type == 'flow' else args.model_config.output_channels)
            xnn.utils.print_yellow("=> output channels is not given - setting to {} - not sure to work".format(args.model_config.output_channels))
        #
        if not isinstance(args.model_config.output_channels,(list,tuple)):
            args.model_config.output_channels = [args.model_config.output_channels]

    if (args.class_weights is None) and ('class_weights' in dir(train_dataset)):
        args.class_weights = train_dataset.class_weights()
        if not isinstance(args.class_weights, (list,tuple)):
            args.class_weights = [args.class_weights]
        #
        print("=> class weights available for dataset: {}".format(args.class_weights))

    #################################################
    pretrained_data = None
    model_surgery_quantize = False
    pretrained_data = None
    if args.pretrained and args.pretrained != "None":
        pretrained_data = []
        pretrained_files = args.pretrained if isinstance(args.pretrained,(list,tuple)) else [args.pretrained]
        for p in pretrained_files:
            if isinstance(p, dict):
                p_data = p
            else:
                if p.startswith('http://') or p.startswith('https://'):
                    p_file = xnn.utils.download_url(p, './data/downloads')
                else:
                    p_file = p
                #
                print(f'=> loading pretrained weights file: {p}')
                p_data = torch.load(p_file, map_location='cpu')
            #
            pretrained_data.append(p_data)
            model_surgery_quantize = p_data['quantize'] if 'quantize' in p_data else False
    #

    #################################################
    # create model
    is_onnx_model = False
    if isinstance(args.model, torch.nn.Module):
        model, change_names_dict = args.model if isinstance(args.model, (list, tuple)) else (args.model, None)
        assert isinstance(model, torch.nn.Module), 'args.model, if provided must be a valid torch.nn.Module'
    elif isinstance(args.model, str) and args.model.endswith('.onnx'):
        model = xnn.onnx.import_onnx(args.model)
        is_onnx_model = True
    else:
        xnn.utils.print_yellow("=> creating model '{}'".format(args.model_name))
        model = xvision.models.pixel2pixel.__dict__[args.model_name](args.model_config)
        # check if we got the model as well as parameters to change the names in pretrained
        model, change_names_dict = model if isinstance(model, (list,tuple)) else (model,None)
    #

    if args.quantize:
        # dummy input is used by quantized models to analyze graph
        is_cuda = next(model.parameters()).is_cuda
        dummy_input = create_rand_inputs(args, is_cuda=is_cuda)
        #
        if 'training' in args.phase:
            model = edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantTrainModule(model, per_channel_q=args.per_channel_q,
                        histogram_range=args.histogram_range, bitwidth_weights=args.bitwidth_weights,
                        bitwidth_activations=args.bitwidth_activations, constrain_bias=args.constrain_bias,
                        dummy_input=dummy_input, total_epochs=args.epochs)
        elif 'calibration' in args.phase:
            model = edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantCalibrateModule(model, per_channel_q=args.per_channel_q,
                        bitwidth_weights=args.bitwidth_weights, bitwidth_activations=args.bitwidth_activations,
                        histogram_range=args.histogram_range, constrain_bias=args.constrain_bias,
                        bias_calibration=args.bias_calibration, dummy_input=dummy_input, lr_calib=args.lr_calib)
        elif 'validation' in args.phase:
            # Note: bias_calibration is not emabled
            model = edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantTestModule(model, per_channel_q=args.per_channel_q,
                        bitwidth_weights=args.bitwidth_weights, bitwidth_activations=args.bitwidth_activations,
                        histogram_range=args.histogram_range, constrain_bias=args.constrain_bias,
                        dummy_input=dummy_input, model_surgery_quantize=model_surgery_quantize)
        else:
            assert False, f'invalid phase {args.phase}'
    #

    # load pretrained model
    if pretrained_data is not None and not is_onnx_model:
        model_orig = get_model_orig(model)
        for (p_data,p_file) in zip(pretrained_data, pretrained_files):
            print("=> using pretrained weights from: {}".format(p_file))
            if hasattr(model_orig, 'load_weights'):
                model_orig.load_weights(pretrained=p_data, change_names_dict=change_names_dict)
            else:
                xnn.utils.load_weights(get_model_orig(model), pretrained=p_data, change_names_dict=change_names_dict)
            #
        #
    #

    #################################################
    if args.count_flops:
        count_flops(args, model)

    #################################################
    if args.save_onnx:
        write_onnx_model(args, get_model_orig(model), save_path, save_traced_model=False)
    #

    #################################################
    if args.print_model:
        print(model)
        print('\n')
    else:
        args.logger.debug(str(model))
        args.logger.debug('\n')

    #################################################
    if (not args.run_soon):
        print("Training not needed for now")
        close(args)
        exit()

    #################################################
    # DataParallel does not work for QuantCalibrateModule or QuantTestModule
    if args.parallel_model and (not isinstance(model, (edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantCalibrateModule, edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantTestModule))):
        model = torch.nn.DataParallel(model)

    #################################################
    model = module_to_device(model)

    #################################################
    # for help in debug/print
    for name, module in model.named_modules():
        module.name = name

    #################################################
    args.loss_modules = copy.deepcopy(args.losses)
    for task_dx, task_losses in enumerate(args.losses):
        for loss_idx, loss_fn in enumerate(task_losses):
            kw_args = {}
            loss_args = pixel2pixel_losses.__dict__[loss_fn].args()
            for arg in loss_args:
                if arg == 'weight' and (args.class_weights is not None):
                    kw_args.update({arg:args.class_weights[task_dx]})
                elif arg == 'num_classes':
                    kw_args.update({arg:args.model_config.output_channels[task_dx]})
                elif arg == 'sparse':
                    kw_args.update({arg:args.sparse})
                elif arg == 'enable_fp16':
                    kw_args.update({arg:args.model_config.enable_fp16})
                #
            #
            loss_fn_raw = pixel2pixel_losses.__dict__[loss_fn](**kw_args)
            if args.parallel_criterion:
                loss_fn = module_to_device(torch.nn.DataParallel(loss_fn_raw)) if args.parallel_criterion else module_to_device(loss_fn_raw)
                loss_fn.info = loss_fn_raw.info
                loss_fn.clear = loss_fn_raw.clear
            else:
                loss_fn = to_device(loss_fn_raw)
            #
            args.loss_modules[task_dx][loss_idx] = loss_fn
    #

    args.metric_modules = copy.deepcopy(args.metrics)
    for task_dx, task_metrics in enumerate(args.metrics):
        for midx, metric_fn in enumerate(task_metrics):
            kw_args = {}
            loss_args = pixel2pixel_losses.__dict__[metric_fn].args()
            for arg in loss_args:
                if arg == 'weight':
                    kw_args.update({arg:args.class_weights[task_dx]})
                elif arg == 'num_classes':
                    kw_args.update({arg:args.model_config.output_channels[task_dx]})
                elif arg == 'sparse':
                    kw_args.update({arg:args.sparse})
                elif arg == 'enable_fp16':
                    kw_args.update({arg:args.model_config.enable_fp16})
                #
            #
            metric_fn_raw = pixel2pixel_losses.__dict__[metric_fn](**kw_args)
            if args.parallel_criterion:
                metric_fn = module_to_device(torch.nn.DataParallel(metric_fn_raw))
                metric_fn.info = metric_fn_raw.info
                metric_fn.clear = metric_fn_raw.clear
            else:
                metric_fn = module_to_device(metric_fn_raw)
            #
            args.metric_modules[task_dx][midx] = metric_fn
    #

    #################################################
    if args.phase=='validation':
        with torch.no_grad():
            validate(args, val_dataset, val_loader, model, 0, val_writer)
        #
        close(args)
        return

    #################################################
    assert(args.optimizer in ['adam', 'sgd'])
    print('=> setting {} optimizer'.format(args.optimizer))
    if args.lr_clips is not None:
        learning_rate_clips = args.lr_clips if 'training' in args.phase else 0.0
        clips_decay = args.bias_decay if (args.bias_decay is not None and args.bias_decay != 0.0) else args.weight_decay
        clips_params = [p for n,p in model.named_parameters() if 'clips' in n]
        other_params = [p for n,p in model.named_parameters() if 'clips' not in n]
        param_groups = [{'params': clips_params, 'weight_decay': clips_decay, 'lr': learning_rate_clips},
                        {'params': other_params, 'weight_decay': args.weight_decay}]
    else:
        param_groups = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'weight_decay': args.weight_decay}]
    #

    learning_rate = args.lr if ('training'in args.phase) else 0.0
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_groups, learning_rate, betas=(args.momentum, args.beta))
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_groups, learning_rate, momentum=args.momentum)
    else:
        raise ValueError('Unknown optimizer type{}'.format(args.optimizer))
    #

    #################################################
    max_iter = args.epochs * len(train_loader)
    scheduler = xnn.optim.lr_scheduler.SchedulerWrapper(scheduler_type=args.scheduler, optimizer=optimizer,
                    epochs=args.epochs, start_epoch=args.start_epoch,
                    warmup_epochs=args.warmup_epochs, warmup_factor=args.warmup_factor,
                    max_iter=max_iter, polystep_power=args.polystep_power,
                    milestones=args.milestones, multistep_gamma=args.multistep_gamma)

    # optionally resume from a checkpoint
    if args.resume:
        if not os.path.isfile(args.resume):
            print("=> no checkpoint found at '{}'".format(args.resume))        
        else:
            print("=> loading checkpoint '{}'".format(args.resume))

        checkpoint = torch.load(args.resume)
        model = xnn.utils.load_weights(model, checkpoint)
            
        if args.start_epoch == 0:
            args.start_epoch = checkpoint['epoch']
        
        if 'best_metric' in list(checkpoint.keys()):    
            args.best_metric = checkpoint['best_metric']

        if 'optimizer' in list(checkpoint.keys()):  
            optimizer.load_state_dict(checkpoint['optimizer'])

        if 'scheduler' in list(checkpoint.keys()):
            scheduler.load_state_dict(checkpoint['scheduler'])

        if 'multi_task_factors' in list(checkpoint.keys()):
            args.multi_task_factors = checkpoint['multi_task_factors']

        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    #################################################
    if args.evaluate_start:
        with torch.no_grad():
            validate(args, val_dataset, val_loader, model, args.start_epoch, val_writer , description="_pretrained")

    grad_scaler = torch.cuda.amp.GradScaler() if args.model_config.enable_fp16 else None

    for epoch in range(args.start_epoch, args.epochs):
        # epoch is needed to seed shuffling in DistributedSampler, every epoch.
        # otherwise seed of 0 is used every epoch, which seems incorrect.
        if train_sampler and isinstance(train_sampler, torch.utils.data.DistributedSampler):
            train_sampler.set_epoch(epoch)
        if val_sampler and isinstance(val_sampler, torch.utils.data.DistributedSampler):
            val_sampler.set_epoch(epoch)

        # train for one epoch
        train(args, train_dataset, train_loader, model, optimizer, epoch, train_writer, scheduler, grad_scaler)

        # evaluate on validation set
        with torch.no_grad():
            val_metric, metric_name = validate(args, val_dataset, val_loader, model, epoch, val_writer)

        if args.best_metric < 0:
            args.best_metric = val_metric

        if "iou" in metric_name.lower() or "acc" in metric_name.lower():
            is_best = val_metric >= args.best_metric
            args.best_metric = max(val_metric, args.best_metric)
        elif "error" in metric_name.lower() or "diff" in metric_name.lower() or "norm" in metric_name.lower() \
                or "loss" in metric_name.lower() or "outlier" in metric_name.lower():
            is_best = val_metric <= args.best_metric
            args.best_metric = min(val_metric, args.best_metric)
        else:
            raise ValueError("Metric is not known. Best model could not be saved.")
        #

        checkpoint_dict = { 'epoch': epoch + 1, 'model_name': args.model_name,
                            'state_dict': get_model_orig(model).state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'best_metric': args.best_metric,
                            'multi_task_factors': args.multi_task_factors,
                            'quantize' : args.quantize}

        save_checkpoint(args, save_path, get_model_orig(model), checkpoint_dict, is_best)

        if args.tensorboard_enable:
            train_writer.file_writer.flush()
            val_writer.file_writer.flush()

        # adjust the learning rate using lr scheduler
        if 'training' in args.phase:
            scheduler.step()
        #
    #

    # close and cleanup
    close(args)
#

###################################################################
def is_valid_phase(phase):
    phases = ('training', 'calibration', 'validation')
    return any(p in phase for p in phases)


###################################################################
def train(args, train_dataset, train_loader, model, optimizer, epoch, train_writer, scheduler, grad_scaler):
    to_device = lambda src_object, non_blocking=False: src_object.cuda(non_blocking=non_blocking) if args.device in ('cuda', None) else src_object
    batch_time = xnn.utils.AverageMeter()
    data_time = xnn.utils.AverageMeter()
    # if the loss/ metric is already an average, no need to further average
    avg_loss = [xnn.utils.AverageMeter(print_avg=(not task_loss[0].info()['is_avg'])) for task_loss in args.loss_modules]
    avg_loss_orig = [xnn.utils.AverageMeter(print_avg=(not task_loss[0].info()['is_avg'])) for task_loss in args.loss_modules]
    avg_metric = [xnn.utils.AverageMeter(print_avg=(not task_metric[0].info()['is_avg'])) for task_metric in args.metric_modules]

    ##########################
    # switch to train mode
    model.train()

    #freeze layers 
    if args.freeze_layers is not None:
        # 'freeze_layer_name' could be part of 'name', i.e. 'name' need not be exact same as 'freeze_layer_name'
        # e.g. freeze_layer_name = 'encoder.0' then all layers like, 'encoder.0.0'  'encoder.0.1' will be frozen
        for freeze_layer_name in args.freeze_layers:
            for name, module in model.named_modules():
                if freeze_layer_name in name:
                    xnn.utils.print_once("Freezing the module : {}".format(name))
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False

    ##########################
    for task_dx, task_losses in enumerate(args.loss_modules):
        for loss_idx, loss_fn in enumerate(task_losses):
            loss_fn.clear()
    for task_dx, task_metrics in enumerate(args.metric_modules):
        for midx, metric_fn in enumerate(task_metrics):
            metric_fn.clear()

    num_iter = len(train_loader)
    progress_bar = progiter.ProgIter(np.arange(num_iter), chunksize=1)
    metric_name = "Metric"
    metric_ctx = [None] * len(args.metric_modules)
    end_time = time.time()
    writer_idx = 0
    last_update_iter = -1

    # change color to yellow for calibration
    progressbar_color = (Fore.YELLOW if (('calibration' in args.phase) or ('training' in args.phase and args.quantize)) else Fore.WHITE)
    print('{}'.format(progressbar_color), end='')

    ##########################
    for iter_id, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end_time)

        lr = scheduler.get_lr()[0]

        input_list = [[to_device(jj) for jj in img] if isinstance(img,(list,tuple)) else  to_device(img) for img in inputs]
        target_list = [to_device(tgt, non_blocking=True) for tgt in targets]
        target_sizes = [tgt.shape for tgt in target_list]
        batch_size_cur = target_sizes[0][0]

        ##########################
        # compute output
        task_outputs = model(input_list)

        task_outputs = task_outputs if isinstance(task_outputs,(list,tuple)) else [task_outputs]
        # upsample output to target resolution
        if args.upsample_mode is not None:
            task_outputs = upsample_tensors(task_outputs, target_sizes, args.upsample_mode)

        if args.model_config.multi_task_type is not None and len(args.model_config.output_channels) > 1:
            args.multi_task_factors, args.multi_task_offsets = xnn.layers.get_loss_scales(model)
        else:
            args.multi_task_factors = None
            args.multi_task_offsets = None

        loss_total, loss_list, loss_names, loss_types, loss_list_orig = \
            compute_task_objectives(args, args.loss_modules, input_list, task_outputs, target_list,
                         task_mults=args.multi_task_factors, task_offsets=args.multi_task_offsets,
                         loss_mult_factors=args.loss_mult_factors)

        if args.print_train_class_iou:
            metric_total, metric_list, metric_names, metric_types, _, confusion_matrix = \
                compute_task_objectives(args, args.metric_modules, input_list, task_outputs, target_list, 
                get_confusion_matrix=args.print_train_class_iou)
        else:        
            metric_total, metric_list, metric_names, metric_types, _ = \
                compute_task_objectives(args, args.metric_modules, input_list, task_outputs, target_list, 
                get_confusion_matrix=args.print_train_class_iou)

        if args.model_config.multi_task_type is not None and len(args.model_config.output_channels) > 1:
            xnn.layers.set_losses(model, loss_list_orig)

        if 'training' in args.phase:
            # accumulate gradients
            if args.model_config.enable_fp16:
                grad_scaler.scale(loss_total).backward()
            else:
                loss_total.backward()
            #

            # optimization step
            if ((iter_id+1) % args.iter_size) == 0:
                if args.model_config.enable_fp16:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                #
                # zero gradients so that we can accumulate gradients
                # setting grad=None is a faster alternative instead of optimizer.zero_grad()
                xnn.utils.clear_grad(model)
            #
        #

        # record loss.
        for task_idx, task_losses in enumerate(args.loss_modules):
            avg_loss[task_idx].update(float(loss_list[task_idx].cpu()), batch_size_cur)
            avg_loss_orig[task_idx].update(float(loss_list_orig[task_idx].cpu()), batch_size_cur)
            if args.tensorboard_enable:
                train_writer.add_scalar('Training/Task{}_{}_Loss_Iter'.format(task_idx,loss_names[task_idx]), float(loss_list[task_idx]), args.n_iter)
                if args.model_config.multi_task_type is not None and len(args.model_config.output_channels) > 1:
                    train_writer.add_scalar('Training/multi_task_Factor_Task{}_{}'.format(task_idx,loss_names[task_idx]), float(args.multi_task_factors[task_idx]), args.n_iter)

        # record error/accuracy.
        for task_idx, task_metrics in enumerate(args.metric_modules):
            avg_metric[task_idx].update(float(metric_list[task_idx].cpu()), batch_size_cur)

        ##########################
        if args.tensorboard_enable:
            write_output(args, 'Training_', num_iter, iter_id, epoch, train_dataset, train_writer, input_list, task_outputs, target_list, metric_name, writer_idx)

        if ((iter_id % args.print_freq) == 0) or (iter_id == (num_iter-1)):
            output_string = ''
            for task_idx, task_metrics in enumerate(args.metric_modules):
                output_string += '[{}={}]'.format(metric_names[task_idx], str(avg_metric[task_idx]))

            epoch_str = '{}/{}'.format(epoch + 1, args.epochs)
            progress_bar.set_description("{}=> {}  ".format(progressbar_color, args.phase))
            multi_task_factors_print = ['{:.3f}'.format(float(lmf)) for lmf in args.multi_task_factors] if args.multi_task_factors is not None else None
            progress_bar.set_postfix(dict(Epoch=epoch_str, LR=lr, DataTime=str(data_time), LossMult=multi_task_factors_print, Loss=avg_loss, Output=output_string))
            progress_bar.update(iter_id-last_update_iter)
            last_update_iter = iter_id

        args.n_iter += 1
        end_time = time.time()
        writer_idx = (writer_idx + 1) % args.tensorboard_num_imgs

        # add onnx graph to tensorboard
        # commenting out due to issues in transitioning to pytorch 0.4
        # (bilinear mode in upsampling causes hang or crash - may be due to align_borders change, nearest is fine)
        #if epoch == 0 and iter_id == 0:
        #    input_zero = torch.zeros(input_var.shape)
        #    train_writer.add_graph(model, input_zero)
        #This cache operation slows down tranining  
        #torch.cuda.empty_cache()
    #

    if args.print_train_class_iou:
        print_class_iou(args=args, confusion_matrix=confusion_matrix, task_idx=task_idx)
        
    progress_bar.close()

    # to print a new line - do not provide end=''
    print('{}'.format(Fore.RESET), end='')

    if args.tensorboard_enable:
        for task_idx, task_losses in enumerate(args.loss_modules):
            train_writer.add_scalar('Training/Task{}_{}_Loss_Epoch'.format(task_idx,loss_names[task_idx]), float(avg_loss[task_idx]), epoch)

        for task_idx, task_metrics in enumerate(args.metric_modules):
            train_writer.add_scalar('Training/Task{}_{}_Metric_Epoch'.format(task_idx,metric_names[task_idx]), float(avg_metric[task_idx]), epoch)

    output_name = metric_names[args.pivot_task_idx]
    output_metric = float(avg_metric[args.pivot_task_idx])

    ##########################
    if args.quantize:
        def debug_format(v):
            return ('{:.3f}'.format(v) if v is not None else 'None')
        #
        clips_act = [m.get_clips_act()[1] for n,m in model.named_modules() if isinstance(m,xnn.layers.PAct2)]
        if len(clips_act) > 0:
            args.logger.debug('\nclips_act : ' + ' '.join(map(debug_format, clips_act)))
            args.logger.debug('')
    #
    return output_metric, output_name


###################################################################
def validate(args, val_dataset, val_loader, model, epoch, val_writer, description=""):
    to_device = lambda src_object, non_blocking=False: src_object.cuda(non_blocking=non_blocking) if args.device in ('cuda', None) else src_object
    data_time = xnn.utils.AverageMeter()
    # if the loss/ metric is already an average, no need to further average
    avg_metric = [xnn.utils.AverageMeter(print_avg=(not task_metric[0].info()['is_avg'])) for task_metric in args.metric_modules]

    ##########################
    # switch to evaluate mode
    model.eval()

    ##########################
    for task_dx, task_metrics in enumerate(args.metric_modules):
        for midx, metric_fn in enumerate(task_metrics):
            metric_fn.clear()

    metric_name = "Metric"
    end_time = time.time()
    writer_idx = 0
    last_update_iter = -1
    metric_ctx = [None] * len(args.metric_modules)

    num_iter = len(val_loader)
    progress_bar = progiter.ProgIter(np.arange(num_iter), chunksize=1)

    # change color to green
    print('{}'.format(Fore.GREEN), end='')

    ##########################
    for iter_id, (inputs, targets) in enumerate(val_loader):
        data_time.update(time.time() - end_time)
        input_list = [[to_device(jj) for jj in img] if isinstance(img,(list,tuple)) else to_device(img) for img in inputs]
        target_list = [to_device(j, non_blocking=True) for j in targets]
        target_sizes = [tgt.shape for tgt in target_list]
        batch_size_cur = target_sizes[0][0]

        # compute output
        task_outputs = model(input_list)

        task_outputs = task_outputs if isinstance(task_outputs, (list, tuple)) else [task_outputs]
        if args.upsample_mode is not None:
           task_outputs = upsample_tensors(task_outputs, target_sizes, args.upsample_mode)
        
        if args.print_val_class_iou:
            metric_total, metric_list, metric_names, metric_types, _, confusion_matrix = \
                compute_task_objectives(args, args.metric_modules, input_list, task_outputs, target_list, 
                get_confusion_matrix = args.print_val_class_iou)
        else:        
            metric_total, metric_list, metric_names, metric_types, _ = \
                compute_task_objectives(args, args.metric_modules, input_list, task_outputs, target_list, 
                get_confusion_matrix = args.print_val_class_iou)

        # record error/accuracy.
        for task_idx, task_metrics in enumerate(args.metric_modules):
            avg_metric[task_idx].update(float(metric_list[task_idx].cpu()), batch_size_cur)

        if args.tensorboard_enable:
            write_output(args, 'Validation_', num_iter, iter_id, epoch, val_dataset, val_writer, input_list, task_outputs, target_list, metric_names, writer_idx)

        if ((iter_id % args.print_freq) == 0) or (iter_id == (num_iter-1)):
            output_string = ''
            for task_idx, task_metrics in enumerate(args.metric_modules):
                output_string += '[{}={}]'.format(metric_names[task_idx], str(avg_metric[task_idx]))

            epoch_str = '{}/{}'.format(epoch + 1, args.epochs)
            progress_bar.set_description("=> validation" + description)
            progress_bar.set_postfix(dict(Epoch=epoch_str, DataTime=data_time, Output="{}".format(output_string)))
            progress_bar.update(iter_id-last_update_iter)
            last_update_iter = iter_id
        #

        end_time = time.time()
        writer_idx = (writer_idx + 1) % args.tensorboard_num_imgs
    #

    if args.print_val_class_iou:
        print_class_iou(args = args, confusion_matrix = confusion_matrix, task_idx=task_idx)
    #

    #print_conf_matrix(conf_matrix=conf_matrix, en=False)
    progress_bar.close()

    # to print a new line - do not provide end=''
    print('{}'.format(Fore.RESET), end='')

    if args.tensorboard_enable:
        for task_idx, task_metrics in enumerate(args.metric_modules):
            val_writer.add_scalar('Validation/Task{}_{}_Metric_Epoch'.format(task_idx,metric_names[task_idx]), float(avg_metric[task_idx]), epoch)

    output_name = metric_names[args.pivot_task_idx]
    output_metric = float(avg_metric[args.pivot_task_idx])
    return output_metric, output_name


###################################################################
def close(args):
    if args.logger is not None:
        args.logger.close()
        del args.logger
        args.logger = None
    #
    args.best_metric = -1
#


def get_save_path(args, phase=None):
    date = args.date if args.date else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join('./data/checkpoints/edgeailite', args.dataset_name, date + '_' + args.dataset_name + '_' + args.model_name)
    save_path += '_resize{}x{}_traincrop{}x{}'.format(args.img_resize[1], args.img_resize[0], args.rand_crop[1], args.rand_crop[0])
    phase = phase if (phase is not None) else args.phase
    save_path = os.path.join(save_path, phase)
    return save_path


def get_model_orig(model):
    is_parallel_model = isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    model_orig = (model.module if is_parallel_model else model)
    model_orig = (model_orig.module if isinstance(model_orig, (edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantBaseModule)) else model_orig)
    return model_orig


def create_rand_inputs(args, is_cuda):
    to_device = lambda src_object, non_blocking=False: src_object.cuda(non_blocking=non_blocking) if args.device in ('cuda', None) else src_object
    dummy_input = []
    if not args.model_config.input_nv12:
        for i_ch in args.model_config.input_channels:
            x = torch.rand((1, i_ch, args.img_resize[0], args.img_resize[1]))
            x = to_device(x) if is_cuda else x
            dummy_input.append(x)
    else: #nv12    
        for i_ch in args.model_config.input_channels:
            y = torch.rand((1, 1, args.img_resize[0], args.img_resize[1]))
            uv = torch.rand((1, 1, args.img_resize[0]//2, args.img_resize[1]))
            y = to_device(y) if is_cuda else y
            uv = to_device(uv) if is_cuda else uv
            dummy_input.append([y,uv])

    return dummy_input


def count_flops(args, model):
    is_cuda = next(model.parameters()).is_cuda
    dummy_input = create_rand_inputs(args, is_cuda)
    model.eval()
    total_mult_adds, total_params = xnn.utils.get_model_complexity(model, dummy_input)
    total_mult_adds_giga = total_mult_adds/1e9
    total_flops = total_mult_adds_giga*2
    total_params_mega = total_params/1e6
    print('=> Resize = {}, GFLOPs = {}, GMACs = {}, MegaParams = {}'.format(args.img_resize, total_flops, total_mult_adds_giga, total_params_mega))


def derive_node_name(input_name):
    #take last entry of input names for deciding node name
    #print("input_name[-1]: ", input_name[-1])
    node_name = input_name[-1].rsplit('.', 1)[0]
    #print("formed node_name: ", node_name)
    return node_name


#torch onnx export does not update names. Do it using onnx.save
def add_node_names(onnx_model_name):
    onnx_model = onnx.load(onnx_model_name)
    for i in range(len(onnx_model.graph.node)):
        for j in range(len(onnx_model.graph.node[i].input)):
            #print('-'*60)
            #print("name: ", onnx_model.graph.node[i].name)
            #print("input: ", onnx_model.graph.node[i].input)
            #print("output: ", onnx_model.graph.node[i].output)

            ## these cause name conflicts in some onnx models
            # onnx_model.graph.node[i].input[j] = onnx_model.graph.node[i].input[j].split(':')[0]
            # print(onnx_model.graph.node[i].input[j])
            onnx_model.graph.node[i].name = derive_node_name(onnx_model.graph.node[i].input) + str(i) # resolve name conflicts
        #
    #
    #update model inplace
    onnx.save(onnx_model, onnx_model_name)


def write_onnx_model(args, model, save_path, name='checkpoint.onnx', save_traced_model=False):
    is_cuda = next(model.parameters()).is_cuda
    input_list = create_rand_inputs(args, is_cuda=is_cuda)
    onnx_file = os.path.join(save_path, name)
    model.eval()
    torch.onnx.export(model, input_list, onnx_file, export_params=True, verbose=False,
                      do_constant_folding=True, opset_version=args.opset_version)

    #torch onnx export does not update names. Do it using onnx.save
    # add_node_names(onnx_model_name=onnx_file)
    # infer shapes
    onnx.shape_inference.infer_shapes_path(onnx_file, onnx_file)

    if save_traced_model:
        traced_model = torch.jit.trace(model, (input_list,))
        traced_save_path = os.path.join(save_path, 'traced_model.pth')
        torch.jit.save(traced_model, traced_save_path)
    #


###################################################################
def write_output(args, prefix, val_epoch_size, iter_id, epoch, dataset, output_writer, input_images, task_outputs, task_targets, metric_names, writer_idx):
    write_freq = (args.tensorboard_num_imgs / float(val_epoch_size))
    write_prob = np.random.random()
    if (write_prob > write_freq):
        return
    if args.model_config.input_nv12:
        batch_size = input_images[0][0].shape[0]
    else:
        batch_size = input_images[0].shape[0]
    b_index = random.randint(0, batch_size - 1)

    input_image = None
    for img_idx, img in enumerate(input_images):
        if args.model_config.input_nv12:
            #convert NV12 to BGR for tensorboard
            input_image = xvision.transforms.image_transforms_xv12.nv12_to_bgr_image(Y = input_images[img_idx][0][b_index], UV = input_images[img_idx][1][b_index],
                                   image_scale=args.image_scale, image_mean=args.image_mean)
        else:
            input_image = input_images[img_idx][b_index].cpu().numpy().transpose((1, 2, 0))
            # convert back to original input range (0-255)
            input_image = input_image / args.image_scale + args.image_mean

        if args.is_flow and args.is_flow[0][img_idx]:
            #input corresponding to flow is assumed to have been generated by adding 128
            flow = input_image - 128
            flow_hsv = xnn.utils.flow2hsv(flow.transpose(2, 0, 1), confidence=False).transpose(2, 0, 1)
            #flow_hsv = (flow_hsv / 255.0).clip(0, 1) #TODO: check this
            output_writer.add_image(prefix +'Input{}/{}'.format(img_idx, writer_idx), flow_hsv, epoch)
        else:
            input_image = (input_image/255.0).clip(0,1) #.astype(np.uint8)
            output_writer.add_image(prefix + 'Input{}/{}'.format(img_idx, writer_idx), input_image.transpose((2,0,1)), epoch)

    # for sparse data, chroma blending does not look good
    for task_idx, output_type in enumerate(args.model_config.output_type):
        # metric_name = metric_names[task_idx]
        output = task_outputs[task_idx]
        target = task_targets[task_idx]
        if (output_type == 'segmentation') and hasattr(dataset, 'decode_segmap'):
            segmentation_target = dataset.decode_segmap(target[b_index,0].cpu().numpy())
            segmentation_output = output.max(dim=1,keepdim=True)[1].data.cpu().numpy() if(output.shape[1]>1) else output.data.cpu().numpy()
            segmentation_output = dataset.decode_segmap(segmentation_output[b_index,0])
            segmentation_output_blend = xnn.utils.chroma_blend(input_image, segmentation_output)
            #
            output_writer.add_image(prefix+'Task{}_{}_GT/{}'.format(task_idx,output_type,writer_idx), segmentation_target.transpose(2,0,1), epoch)
            if not args.sparse:
                segmentation_target_blend = xnn.utils.chroma_blend(input_image, segmentation_target)
                output_writer.add_image(prefix + 'Task{}_{}_GT_ColorBlend/{}'.format(task_idx, output_type, writer_idx), segmentation_target_blend.transpose(2, 0, 1), epoch)
            #
            output_writer.add_image(prefix+'Task{}_{}_Output/{}'.format(task_idx,output_type,writer_idx), segmentation_output.transpose(2,0,1), epoch)
            output_writer.add_image(prefix+'Task{}_{}_Output_ColorBlend/{}'.format(task_idx,output_type,writer_idx), segmentation_output_blend.transpose(2,0,1), epoch)
        elif (output_type in ('depth', 'disparity')):
            depth_chanidx = 0
            output_writer.add_image(prefix+'Task{}_{}_GT_Color_Visualization/{}'.format(task_idx,output_type,writer_idx), xnn.utils.tensor2array(target[b_index][depth_chanidx].cpu(), max_value=args.max_depth, colormap=args.viz_colormap).transpose(2,0,1), epoch)
            if not args.sparse:
                output_writer.add_image(prefix + 'Task{}_{}_GT_ColorBlend_Visualization/{}'.format(task_idx, output_type, writer_idx), xnn.utils.tensor2array(target[b_index][depth_chanidx].cpu(), max_value=args.max_depth, colormap=args.viz_colormap, input_blend=input_image).transpose(2, 0, 1), epoch)
            #
            output_writer.add_image(prefix+'Task{}_{}_Output_Color_Visualization/{}'.format(task_idx,output_type,writer_idx), xnn.utils.tensor2array(output.data[b_index][depth_chanidx].cpu(), max_value=args.max_depth, colormap=args.viz_colormap).transpose(2,0,1), epoch)
            output_writer.add_image(prefix + 'Task{}_{}_Output_ColorBlend_Visualization/{}'.format(task_idx, output_type, writer_idx),xnn.utils.tensor2array(output.data[b_index][depth_chanidx].cpu(), max_value=args.max_depth, colormap=args.viz_colormap, input_blend=input_image).transpose(2, 0, 1), epoch)
        elif (output_type == 'flow'):
            max_value_flow = 10.0 # only for visualization
            output_writer.add_image(prefix+'Task{}_{}_GT/{}'.format(task_idx,output_type,writer_idx), xnn.utils.flow2hsv(target[b_index][:2].cpu().numpy(), max_value=max_value_flow).transpose(2,0,1), epoch)
            output_writer.add_image(prefix+'Task{}_{}_Output/{}'.format(task_idx,output_type,writer_idx), xnn.utils.flow2hsv(output.data[b_index][:2].cpu().numpy(), max_value=max_value_flow).transpose(2,0,1), epoch)
        elif (output_type == 'interest_pt'):
            score_chanidx = 0
            target_score_to_write = target[b_index][score_chanidx].cpu()
            output_score_to_write = output.data[b_index][score_chanidx].cpu()
            
            #if score is learnt as zero mean add offset to make it [0-255]
            if args.make_score_zero_mean:
                # target_score_to_write!=0 : value 0 indicates GT unavailble. Leave them to be 0.
                target_score_to_write[target_score_to_write!=0] += 128.0
                output_score_to_write += 128.0

            max_value_score = float(torch.max(target_score_to_write)) #0.002
            output_writer.add_image(prefix+'Task{}_{}_GT_Bone_Visualization/{}'.format(task_idx,output_type,writer_idx), xnn.utils.tensor2array(target_score_to_write, max_value=max_value_score, colormap='bone').transpose(2,0,1), epoch)
            output_writer.add_image(prefix+'Task{}_{}_Output_Bone_Visualization/{}'.format(task_idx,output_type,writer_idx), xnn.utils.tensor2array(output_score_to_write, max_value=max_value_score, colormap='bone').transpose(2,0,1), epoch)
        #

def print_conf_matrix(conf_matrix = [], en = False):
    if not en:
        return
    num_rows = conf_matrix.shape[0]
    num_cols = conf_matrix.shape[1]
    print("-"*64)
    num_ele = 1
    for r_idx in range(num_rows):
        print("\n")
        for c_idx in range(0,num_cols,num_ele):
            print(conf_matrix[r_idx][c_idx:c_idx+num_ele], end="")
    print("\n")
    print("-" * 64)

def compute_task_objectives(args, objective_fns, input_var, task_outputs, task_targets, task_mults=None, 
  task_offsets=None, loss_mult_factors=None, get_confusion_matrix = False):
  
    ##########################
    objective_total = torch.zeros_like(task_outputs[0].view(-1)[0])
    objective_list = []
    objective_list_orig = []
    objective_names = []
    objective_types = []
    for task_idx, task_objectives in enumerate(objective_fns):
        output_type = args.model_config.output_type[task_idx]
        objective_sum_value = torch.zeros_like(task_outputs[task_idx].view(-1)[0])
        objective_sum_name = ''
        objective_sum_type = ''

        task_mult = task_mults[task_idx] if task_mults is not None else 1.0
        task_offset = task_offsets[task_idx] if task_offsets is not None else 0.0

        for oidx, objective_fn in enumerate(task_objectives):
            objective_batch = objective_fn(input_var, task_outputs[task_idx], task_targets[task_idx])
            objective_batch = objective_batch.mean() if isinstance(objective_fn, torch.nn.DataParallel) else objective_batch
            objective_name = objective_fn.info()['name']
            objective_type = objective_fn.info()['is_avg']
            if get_confusion_matrix:
                confusion_matrix = objective_fn.info()['confusion_matrix']

            loss_mult = loss_mult_factors[task_idx][oidx] if (loss_mult_factors is not None) else 1.0
            # --
            objective_batch_not_nan = (objective_batch if not torch.isnan(objective_batch) else 0.0)
            objective_sum_value = objective_batch_not_nan*loss_mult + objective_sum_value
            objective_sum_name += (objective_name if (objective_sum_name == '') else ('+' + objective_name))
            assert (objective_sum_type == '' or objective_sum_type == objective_type), 'metric types (avg/val) for a given task should match'
            objective_sum_type = objective_type

        objective_list.append(objective_sum_value)
        objective_list_orig.append(objective_sum_value)
        objective_names.append(objective_sum_name)
        objective_types.append(objective_sum_type)

        objective_total = objective_sum_value*task_mult + task_offset + objective_total

    return_list = [objective_total, objective_list, objective_names, objective_types, objective_list_orig]
    if get_confusion_matrix:
        return_list.append(confusion_matrix)

    return return_list 


def save_checkpoint(args, save_path, model, checkpoint_dict, is_best, filename='checkpoint.pth'):
    torch.save(checkpoint_dict, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth'))
    #
    if args.save_onnx:
        write_onnx_model(args, model, save_path, name='checkpoint.onnx')
        if is_best:
            write_onnx_model(args, model, save_path, name='model_best.onnx')
    #


def get_dataset_sampler(dataset_object, epoch_size):
    print('=> creating a random sampler as epoch_size is specified')
    num_samples = len(dataset_object)
    epoch_size = int(epoch_size * num_samples) if epoch_size < 1 else int(epoch_size)
    dataset_sampler = torch.utils.data.sampler.RandomSampler(data_source=dataset_object, replacement=True, num_samples=epoch_size)
    return dataset_sampler


def get_train_transform(args):
    # image normalization can be at the beginning of transforms or at the end
    image_mean = np.array(args.image_mean, dtype=np.float32)
    image_scale = np.array(args.image_scale, dtype=np.float32)
    image_prenorm = image_transforms.NormalizeMeanScale(mean=image_mean, scale=image_scale) if args.image_prenorm else image_transforms.BypassImages()
    image_postnorm = image_transforms.NormalizeMeanScale(mean=image_mean, scale=image_scale) if (not image_prenorm) else image_transforms.BypassImages()
    reverse_channels = image_transforms.ReverseImageChannels() if args.input_channel_reverse else image_transforms.BypassImages()
    color_2_gray = image_transforms.RandomColor2Gray(is_flow=args.is_flow, random_threshold=args.prob_color_to_gray[0]) if args.prob_color_to_gray[0] != 0.0 else image_transforms.BypassImages()

    # crop size used only for training
    image_train_output_scaling = image_transforms.Scale(args.rand_resize, target_size=args.rand_output_size, is_flow=args.is_flow) \
        if (args.rand_output_size is not None and args.rand_output_size != args.rand_resize) else image_transforms.BypassImages()
    train_transform = image_transforms.Compose([
        reverse_channels,
        image_prenorm,
        image_transforms.AlignImages(interpolation=args.interpolation),
        image_transforms.MaskTarget(args.target_mask, 0),
        image_transforms.CropRect(args.img_border_crop),
        image_transforms.RandomRotate(args.transform_rotation, is_flow=args.is_flow) if args.transform_rotation else None,
        image_transforms.RandomScaleCrop(args.rand_resize, scale_range=args.rand_scale, is_flow=args.is_flow, interpolation=args.interpolation),
        image_transforms.RandomHorizontalFlip(is_flow=args.is_flow),
        image_transforms.RandomCrop(args.rand_crop),
        color_2_gray,
        image_train_output_scaling,
        image_postnorm,
        image_transforms.ConvertToTensor()
        ])
    return train_transform


def get_validation_transform(args):
    # image normalization can be at the beginning of transforms or at the end
    image_mean = np.array(args.image_mean, dtype=np.float32)
    image_scale = np.array(args.image_scale, dtype=np.float32)
    image_prenorm = image_transforms.NormalizeMeanScale(mean=image_mean, scale=image_scale) if args.image_prenorm else image_transforms.BypassImages()
    image_postnorm = image_transforms.NormalizeMeanScale(mean=image_mean, scale=image_scale) if (not image_prenorm) else image_transforms.BypassImages()
    reverse_channels = image_transforms.ReverseImageChannels() if args.input_channel_reverse else image_transforms.BypassImages()
    color_2_gray = image_transforms.RandomColor2Gray(is_flow=args.is_flow, random_threshold=args.prob_color_to_gray[1]) if args.prob_color_to_gray[1] != 0.0 else image_transforms.BypassImages()

    # prediction is resized to output_size before evaluation.
    val_transform = image_transforms.Compose([
        reverse_channels,
        image_prenorm,
        image_transforms.AlignImages(interpolation=args.interpolation),
        image_transforms.MaskTarget(args.target_mask, 0),
        image_transforms.CropRect(args.img_border_crop),
        image_transforms.Scale(args.img_resize, target_size=args.output_size, is_flow=args.is_flow, interpolation=args.interpolation),
        color_2_gray,
        image_postnorm,
        image_transforms.ConvertToTensor()
        ])
    return val_transform


def get_transforms(args):
    # Provision to train with val transform - provide rand_scale as (0, 0)
    # Fixing the train-test resolution discrepancy, https://arxiv.org/abs/1906.06423
    always_use_val_transform = (args.rand_scale[0] == 0)
    train_transform = get_validation_transform(args) if always_use_val_transform else get_train_transform(args)
    val_transform = get_validation_transform(args)
    return train_transform, val_transform


def _upsample_impl(tensor, output_size, upsample_mode):
    # upsample of long tensor is not supported currently. covert to float, just to avoid error.
    # we can do thsi only in the case of nearest mode, otherwise output will have invalid values.
    convert_to_float = False
    if isinstance(tensor, (torch.LongTensor,torch.cuda.LongTensor)):
        convert_to_float = True
        original_dtype = tensor.dtype
        tensor = tensor.float()
        upsample_mode = 'nearest'

    dim_added = False
    if len(tensor.shape) < 4:
        tensor = tensor[np.newaxis,...]
        dim_added = True

    if (tensor.size()[-2:] != output_size):
        tensor = torch.nn.functional.interpolate(tensor, output_size, mode=upsample_mode)

    if dim_added:
        tensor = tensor[0,...]

    if convert_to_float:
        tensor = tensor.long() #tensor.astype(original_dtype)

    return tensor


def upsample_tensors(tensors, output_sizes, upsample_mode):
    if isinstance(tensors, (list,tuple)):
        for tidx, tensor in enumerate(tensors):
            tensors[tidx] = _upsample_impl(tensor, output_sizes[tidx][-2:], upsample_mode)
        #
    else:
        tensors = _upsample_impl(tensors, output_sizes[0][-2:], upsample_mode)
    return tensors

#print IoU for each class
def print_class_iou(args = None, confusion_matrix = None, task_idx = 0):    
    n_classes = args.model_config.output_channels[task_idx]
    [accuracy, mean_iou, iou, f1_score] = compute_accuracy(args, confusion_matrix, n_classes)
    print("\n Class IoU: [", end = "")
    for class_iou in iou:
        print("{:0.3f}".format(class_iou), end=",")
    print("]")    

if __name__ == '__main__':
    train_args = get_config()
    main(train_args)
