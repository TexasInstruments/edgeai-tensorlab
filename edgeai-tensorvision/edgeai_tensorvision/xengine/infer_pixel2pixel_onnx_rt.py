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
import time
import sys
import math
import copy
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import datetime
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

import onnx
import onnxruntime

from onnx import helper


from edgeai_torchmodelopt import xnn
from edgeai_tensorvision import xvision

#sys.path.insert(0, '../devkit-datasets/TI/')
#from fisheye_calib import r_fish_to_theta_rect

# ################################################
def get_config():
    args = xnn.utils.ConfigNode()

    args.dataset_config = xnn.utils.ConfigNode()
    args.dataset_config.split_name = 'val'
    args.dataset_config.max_depth_bfr_scaling = 80
    args.dataset_config.depth_scale = 1
    args.dataset_config.train_depth_log = 1
    args.use_semseg_for_depth = False

    args.model_config = xnn.utils.ConfigNode()
    args.model_name = 'deeplabv2lite_mobilenetv2'       # model architecture, overwritten if pretrained is specified
    args.dataset_name = 'flying_chairs'              # dataset type

    args.data_path = './data/datasets'                       # path to dataset
    args.save_path = None            # checkpoints save path
    args.pretrained = None

    args.model_config.output_type = ['flow']                # the network is used to predict flow or depth or sceneflow')
    args.model_config.output_channels = None                 # number of output channels
    args.model_config.input_channels = None                  # number of input channels
    args.model_config.num_classes = None                       # number of classes (for segmentation)
    args.model_config.output_range = None  # max range of output

    args.model_config.num_decoders = None               # number of decoders to use. [options: 0, 1, None]
    args.sky_dir = False

    args.log_file = None                                # log file name
    args.logger = None                          # logger stream to output into

    args.split_file = None                      # train_val split file
    args.split_files = None                     # split list files. eg: train.txt val.txt
    args.split_value = 0.8                      # test_val split proportion (between 0 (only test) and 1 (only train))

    args.workers = 8                            # number of data loading workers

    args.epoch_size = 0                         # manual epoch size (will match dataset size if not specified)
    args.epoch_size_val = 0                     # manual epoch size (will match dataset size if not specified)
    args.batch_size = 8                         # mini_batch_size
    args.total_batch_size = None                # accumulated batch size. total_batch_size = batch_size*iter_size
    args.iter_size = 1                          # iteration size. total_batch_size = batch_size*iter_size

    args.tensorboard_num_imgs = 5               # number of imgs to display in tensorboard
    args.phase = 'validation'                        # evaluate model on validation set
    args.pretrained = None                      # path to pre_trained model
    args.date = None                            # don\'t append date timestamp to folder
    args.print_freq = 10                        # print frequency (default: 100)

    args.div_flow = 1.0                         # value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results
    args.losses = ['supervised_loss']           # loss functions to minimize
    args.metrics = ['supervised_error']         # metric/measurement/error functions for train/validation
    args.class_weights = None                   # class weights

    args.multistep_gamma = 0.5                  # steps for step scheduler
    args.polystep_power = 1.0                   # power for polynomial scheduler

    args.rand_seed = 1                          # random seed
    args.img_border_crop = None                 # image border crop rectangle. can be relative or absolute
    args.target_mask = None                      # mask rectangle. can be relative or absolute. last value is the mask value
    args.img_resize = None                      # image size to be resized to
    args.rand_scale = (1,1.25)                  # random scale range for training
    args.rand_crop = None                       # image size to be cropped to')
    args.output_size = None                     # target output size to be resized to')

    args.count_flops = True                     # count flops and report

    args.shuffle = True                         # shuffle or not
    args.is_flow = None                         # whether entries in images and targets lists are optical flow or not

    args.multi_decoder = True                   # whether to use multiple decoders or unified decoder

    args.create_video = False                   # whether to create video out of the inferred images

    args.input_tensor_name = ['0']              # list of input tensore names

    args.upsample_mode = 'nearest'              # upsample mode to use., choices=['nearest','bilinear']

    args.image_prenorm = True                   # whether normalization is done before all other the transforms
    args.image_mean = [128.0]                   # image mean for input image normalization
    args.image_scale = [1.0/(0.25*256)]         # image scaling/mult for input iamge normalization
    args.quantize = False                       # apply quantized inference or not
    #args.model_surgery = None                   # replace activations with PAct2 activation module. Helpful in quantized training.
    args.bitwidth_weights = 8                   # bitwidth for weights
    args.bitwidth_activations = 8               # bitwidth for activations
    args.histogram_range = True                 # histogram range for calibration
    args.per_channel_q = False                  # apply separate quantizion factor for each channel in depthwise or not
    args.bias_calibration = False                # apply bias correction during quantized inference calibration

    args.frame_IOU = False                      # Print mIOU for each frame
    args.make_score_zero_mean = False           #to make score and desc zero mean
    args.learn_scaled_values_interest_pt = True
    args.save_mod_files = False                 # saves modified files after last commit. Also  stores commit id.
    args.gpu_mode = True                        #False will make inference run on CPU
    args.write_layer_ip_op= False               #True will make it tap inputs outputs for layers
    args.file_format = 'none'                   #Ip/Op tapped points for each layer: None : it will not be written but print will still appear
    args.save_onnx = True
    args.remove_ignore_lbls_in_pred = False     #True: if in the pred where GT has ignore label do not visualize for GT visualization
    args.do_pred_cordi_f2r = False              #true: Do f2r operation on detected location for interet point task
    args.depth_cmap_plasma = False      
    args.visualize_gt = False                   #to vis pred or GT
    args.viz_depth_color_type = 'plasma'        #color type for dpeth visualization
    args.depth = [False]
    args.dump_layers = True

    args.opset_version = 9                      # onnx opset version
    return args


# ################################################
# to avoid hangs in data loader with multi threads
# this was observed after using cv2 image processing functions
# https://github.com/pytorch/pytorch/issues/1355
cv2.setNumThreads(0)

##################################################
np.set_printoptions(precision=3)

# ################################################
def main(args):

    assert not hasattr(args, 'model_surgery'), 'the argument model_surgery is deprecated, it is not needed now - remove it'

    #################################################
    # global settings. rand seeds for repeatability
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    ################################
    # args check and config
    if args.iter_size != 1 and args.total_batch_size is not None:
        warnings.warn("only one of --iter_size or --total_batch_size must be set")
    #
    if args.total_batch_size is not None:
        args.iter_size = args.total_batch_size//args.batch_size
    else:
        args.total_batch_size = args.batch_size*args.iter_size
    #

    assert args.pretrained is not None, 'pretrained path must be provided'

    # onnx generation is filing for post quantized module
    # args.save_onnx = False if (args.quantize) else args.save_onnx
    #################################################
    # set some global flags and initializations
    # keep it in args for now - although they don't belong here strictly
    # using pin_memory is seen to cause issues, especially when when lot of memory is used.
    args.use_pinned_memory = False
    args.n_iter = 0
    args.best_metric = -1

    #################################################
    if args.save_path is None:
        save_path = get_save_path(args)
    else:
        save_path = args.save_path
    #
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #################################################
    if args.log_file:
        log_file = os.path.splitext(os.path.basename(__file__))[0] + '.log'
        args.logger = xnn.utils.TeeLogger(filename=os.path.join(save_path,log_file))

    ################################
    # print everything for log
    print('=> args: ', args)

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

    transforms = get_transforms(args)

    print("=> fetching img pairs in '{}'".format(args.data_path))
    split_arg = args.split_file if args.split_file else (args.split_files if args.split_files else args.split_value)

    val_dataset = xvision.datasets.pixel2pixel.__dict__[args.dataset_name](args.dataset_config, args.data_path, split=split_arg, transforms=transforms)

    print('=> {} val samples found'.format(len(val_dataset)))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=args.use_pinned_memory, shuffle=args.shuffle)

    #################################################
    if (args.model_config.input_channels is None):
        args.model_config.input_channels = (3,)
        print("=> input channels is not given - setting to {}".format(args.model_config.input_channels))

    if (args.model_config.output_channels is None):
        if ('num_classes' in dir(val_dataset)):
            args.model_config.output_channels = val_dataset.num_classes()
        else:
            args.model_config.output_channels = (2 if args.model_config.output_type == 'flow' else args.model_config.output_channels)
            xnn.utils.print_yellow("=> output channels is not given - setting to {} - not sure to work".format(args.model_config.output_channels))
        #
        if not isinstance(args.model_config.output_channels,(list,tuple)):
            args.model_config.output_channels = [args.model_config.output_channels]

    #################################################
    #Load the ONNX model
    onnx_model = onnx.load(args.pretrained)
    try:
        #Check that the IR is well formed
        onnx.checker.check_model(onnx_model)
    except:
        print("ONNX model check failed: IR(Intermediate Representation) is not well formed")

    if args.dump_layers: #add intermediate outputs to the onnx model
        intermediate_layer_value_info = helper.ValueInfoProto()
        intermediate_layer_value_info.name = ''
        for i in range(len(onnx_model.graph.node)):
            for j in range(len(onnx_model.graph.node[i].output)):
                print('-' * 60)
                print("Node:", i, "output_node:", j, onnx_model.graph.node[i].op_type, onnx_model.graph.node[i].output)
                # add each intermediate layer one by one
                if (onnx_model.graph.node[i].op_type == 'Relu') | (onnx_model.graph.node[i].op_type == 'Add') | \
                (onnx_model.graph.node[i].op_type == 'Concat') | (onnx_model.graph.node[i].op_type == 'Resize') | \
                (onnx_model.graph.node[i].op_type == 'Upsample'):
                    intermediate_layer_value_info.name = onnx_model.graph.node[i].output[0]
                    onnx_model.graph.output.append(intermediate_layer_value_info)
        onnx.save(onnx_model, os.path.join(save_path, 'model_mod.onnx'))
        args.pretrained = os.path.join(save_path, 'model_mod.onnx')

    #################################################
    args.loss_modules = copy.deepcopy(args.losses)
    for task_dx, task_losses in enumerate(args.losses):
        for loss_idx, loss_fn in enumerate(task_losses):
            kw_args = {}
            loss_args = xvision.losses.__dict__[loss_fn].args()
            for arg in loss_args:
                #if arg == 'weight':
                #    kw_args.update({arg:args.class_weights[task_dx]})
                if arg == 'num_classes':
                    kw_args.update({arg:args.model_config.output_channels[task_dx]})
                elif arg == 'sparse':
                    kw_args.update({arg:args.sparse})
                #
            #
            loss_fn = xvision.losses.__dict__[loss_fn](**kw_args)
            loss_fn = loss_fn.cuda()
            args.loss_modules[task_dx][loss_idx] = loss_fn

    args.metric_modules = copy.deepcopy(args.metrics)
    for task_dx, task_metrics in enumerate(args.metrics):
        for midx, metric_fn in enumerate(task_metrics):
            kw_args = {}
            loss_args = xvision.losses.__dict__[metric_fn].args()
            for arg in loss_args:
                if arg == 'weight':
                    kw_args.update({arg:args.class_weights[task_dx]})
                elif arg == 'num_classes':
                    kw_args.update({arg:args.model_config.output_channels[task_dx]})
                elif arg == 'sparse':
                    kw_args.update({arg:args.sparse})
                #
            #
            metric_fn = xvision.losses.__dict__[metric_fn](**kw_args)
            metric_fn = metric_fn.cuda()
            args.metric_modules[task_dx][midx] = metric_fn

    #################################################
    if args.palette:
        print('Creating palette')
        args.palette = val_dataset.create_palette()
        for i, p in enumerate(args.palette):
            args.palette[i] = np.array(p, dtype = np.uint8)
            args.palette[i] = args.palette[i][..., ::-1]  # RGB->BGR, since palette is expected to be given in RGB format

    infer_path = []
    for i, p in enumerate(args.model_config.output_channels):
        infer_path.append(os.path.join(save_path, 'Task{}'.format(i)))
        if not os.path.exists(infer_path[i]):
            os.makedirs(infer_path[i])

    #################################################
    with torch.no_grad():
        validate(args, val_dataset, val_loader, onnx_model, 0, infer_path)

    if args.create_video:
        create_video(args, infer_path=infer_path)

    if args.logger is not None:
        args.logger.close()
        args.logger = None


def validate(args, val_dataset, val_loader, model, epoch, infer_path):
    data_time = xnn.utils.AverageMeter()
    avg_metric = [xnn.utils.AverageMeter(print_avg=(not task_metric[0].info()['is_avg'])) for task_metric in args.metric_modules]

    # switch to evaluate mode
    # model.eval()

    session = onnxruntime.InferenceSession(args.pretrained, None)
    input_name = session.get_inputs()[0].name
    input_details = session.get_inputs()
    output_details = session.get_outputs()

    metric_name = "Metric"
    end_time = time.time()
    writer_idx = 0
    last_update_iter = -1
    metric_ctx = [None] * len(args.metric_modules)

    confusion_matrix = []
    for n_cls in args.model_config.output_channels:
        confusion_matrix.append(np.zeros((n_cls, n_cls+1)))
    metric_txt = []
    ard_err = None
    for iter, (input_list, target_list, input_path, target_path) in enumerate(val_loader):
        file_name =  input_path[-1][0]
        print("started inference of file_name:", file_name)
        data_time.update(time.time() - end_time)

        outputs = session.run([], {input_name: input_list[0].cpu().numpy()})
        if args.dump_layers:
            dst_dir = os.path.join(*infer_path[0].split('/')[:-1],'layers_dump' ,"{:04d}".format(args.batch_size*iter))
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            for idx, output in enumerate(outputs):
                onnx_idx = model.graph.output[idx].name
                dst_file = os.path.join(dst_dir,  "{:04d}".format(int(onnx_idx)) + '.bin')
                output.tofile(dst_file)
            outputs = [outputs[0]]  # First element from the model is the final output

        outputs = [torch.tensor(outputs[index]) for index in range(len(outputs))]
        if args.output_size is not None and target_list:
           target_sizes = [tgt.shape for tgt in target_list]
           outputs = upsample_tensors(outputs, target_sizes, args.upsample_mode)
        elif args.output_size is not None and not target_list:
           target_sizes = [args.output_size for _ in range(len(outputs))]
           outputs = upsample_tensors(outputs, target_sizes, args.upsample_mode)
        outputs = [out.cpu() for out in outputs]

        for task_index in range(len(outputs)):
            output = outputs[task_index]
            gt_target = target_list[task_index] if target_list else None
            if args.visualize_gt and target_list:
                if args.model_config.output_type[task_index] is 'depth':
                    output = gt_target
                else:
                    output = gt_target.to(dtype=torch.int8)
                
            if args.remove_ignore_lbls_in_pred and not (args.model_config.output_type[task_index] is 'depth') and target_list :
                output[gt_target == 255] = args.palette[task_index-1].shape[0]-1
            for index in range(output.shape[0]):
                if args.frame_IOU:
                    confusion_matrix[task_index] = np.zeros((args.model_config.output_channels[task_index], args.model_config.output_channels[task_index] + 1))
                prediction = np.array(output[index])
                if output.shape[1]>1:
                    prediction = np.argmax(prediction, axis=0)
                #
                prediction = np.squeeze(prediction)

                if target_list:
                    label = np.squeeze(np.array(target_list[task_index][index]))
                    if not args.model_config.output_type[task_index] is 'depth':
                        confusion_matrix[task_index] = eval_output(args, prediction, label, confusion_matrix[task_index], args.model_config.output_channels[task_index])
                        accuracy, mean_iou, iou, f1_score= compute_accuracy(args, confusion_matrix[task_index], args.model_config.output_channels[task_index])
                        temp_txt = []
                        temp_txt.append(input_path[-1][index])
                        temp_txt.extend(iou)
                        metric_txt.append(temp_txt)
                        print('{}/{} Inferred Frame {} mean_iou={},'.format((args.batch_size*iter+index+1), len(val_dataset), input_path[-1][index], mean_iou))
                        if index == output.shape[0]-1:
                            print('Task={},\npixel_accuracy={},\nmean_iou={},\niou={},\nf1_score = {}'.format(task_index, accuracy, mean_iou, iou, f1_score))
                            sys.stdout.flush()
                    elif args.model_config.output_type[task_index] is 'depth':
                        valid = (label != 0)
                        gt = torch.tensor(label[valid]).float()
                        inference = torch.tensor(prediction[valid]).float()
                        if len(gt) > 2:
                            if ard_err is None:
                                ard_err = [absreldiff_rng3to80(inference, gt).mean()]
                            else:
                                ard_err.append(absreldiff_rng3to80(inference, gt).mean())
                        elif len(gt) < 2:
                            if ard_err is None:
                                ard_err = [0.0]
                            else:
                                ard_err.append(0.0)

                        print('{}/{} ARD: {}'.format((args.batch_size * iter + index), len(val_dataset),torch.tensor(ard_err).mean()))

                seq = input_path[-1][index].split('/')[-4]
                base_file = os.path.basename(input_path[-1][index])

                if args.label_infer:
                    output_image = prediction
                    output_name = os.path.join(infer_path[task_index], seq + '_' + input_path[-1][index].split('/')[-3] + '_' + base_file)
                    cv2.imwrite(output_name, output_image)
                    print('{}/{}'.format((args.batch_size*iter+index), len(val_dataset)))

                if hasattr(args, 'interest_pt') and args.interest_pt[task_index]:
                    print('{}/{}'.format((args.batch_size * iter + index), len(val_dataset)))
                    output_name = os.path.join(infer_path[task_index], seq + '_' + input_path[-1][index].split('/')[-3] + '_' + base_file)
                    output_name_short = os.path.join(infer_path[task_index], os.path.basename(input_path[-1][index]))
                    wrapper_write_desc(args=args, task_index=task_index, outputs=outputs, index=index, output_name=output_name, output_name_short=output_name_short)
                    
                if args.model_config.output_type[task_index] is 'depth':
                    output_name = os.path.join(infer_path[task_index], seq + '_' + input_path[-1][index].split('/')[-3] + '_' + base_file)
                    viz_depth(prediction = prediction, args=args, output_name = output_name, input_name=input_path[-1][task_index])
                    print('{}/{}'.format((args.batch_size * iter + index), len(val_dataset)))

                if args.blend[task_index]:
                    prediction_size = (prediction.shape[0], prediction.shape[1], 3)
                    output_image = args.palette[task_index-1][prediction.ravel()].reshape(prediction_size)
                    input_bgr = cv2.imread(input_path[-1][index]) #Read the actual RGB image
                    input_bgr = cv2.resize(input_bgr, dsize=(prediction.shape[1],prediction.shape[0]))
                    output_image = xnn.utils.chroma_blend(input_bgr, output_image)
                    output_name = os.path.join(infer_path[task_index], input_path[-1][index].split('/')[-4] + '_' + input_path[-1][index].split('/')[-3] + '_' +os.path.basename(input_path[-1][index]))
                    cv2.imwrite(output_name, output_image)
                    print('{}/{}'.format((args.batch_size*iter+index), len(val_dataset)))
                #

                if args.car_mask:   # generating car_mask (required for localization)
                    car_mask = np.logical_or(prediction == 13, prediction == 14, prediction == 16, prediction == 17)
                    prediction[car_mask] = 255
                    prediction[np.invert(car_mask)] = 0
                    output_image = prediction
                    output_name = os.path.join(infer_path[task_index], os.path.basename(input_path[-1][index]))
                    cv2.imwrite(output_name, output_image)
    np.savetxt('metric.txt', metric_txt, fmt='%s')


###############################################################
def get_save_path(args, phase=None):
    date = args.date if args.date else datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_path = os.path.join('./data/checkpoints/edgeailite', args.dataset_name, date + '_' + args.dataset_name + '_' + args.model_name)
    save_path += '_resize{}x{}'.format(args.img_resize[1], args.img_resize[0])
    if args.rand_crop:
        save_path += '_crop{}x{}'.format(args.rand_crop[1], args.rand_crop[0])
    #
    phase = phase if (phase is not None) else args.phase
    save_path = os.path.join(save_path, phase)
    return save_path


def get_model_orig(model):
    is_parallel_model = isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    model_orig = (model.module if is_parallel_model else model)
    model_orig = (model_orig.module if isinstance(model_orig, (edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantBaseModule)) else model_orig)
    return model_orig


def create_rand_inputs(args, is_cuda):
    dummy_input = []
    for i_ch in args.model_config.input_channels:
        x = torch.rand((1, i_ch, args.img_resize[0], args.img_resize[1]))
        x = x.cuda() if is_cuda else x
        dummy_input.append(x)
    #
    return dummy_input


# FIX_ME:SN move to utils
def store_desc(args=[], output_name=[], write_dense=False, desc_tensor=[], prediction=[],
               scale_to_write_kp_loc_to_orig_res=[1.0, 1.0],
               learn_scaled_values=True):
    sys.path.insert(0, './scripts/')
    import write_desc as write_desc

    if args.write_desc_type != 'NONE':
        txt_file_name = output_name.replace(".png", ".txt")
        if write_dense:
            # write desc
            desc_tensor = desc_tensor.astype(np.int16)
            print("writing dense desc(64 ch) op: {} : {} : {} : {}".format(desc_tensor.shape, desc_tensor.dtype,
                                                                           desc_tensor.min(), desc_tensor.max()))
            desc_tensor_name = output_name.replace(".png", "_desc.npy")
            np.save(desc_tensor_name, desc_tensor)

            # utils_hist.comp_hist_tensor3d(x=desc_tensor, name='desc_64ch', en=True, dir='desc_64ch', log=True, ch_dim=0)

            # write score channel
            prediction = prediction.astype(np.int16)

            print("writing dense score ch op: {} : {} : {} : {}".format(prediction.shape, prediction.dtype,
                                                                        prediction.min(),
                                                                        prediction.max()))
            score_tensor_name = output_name.replace(".png", "_score.npy")
            np.save(score_tensor_name, prediction)

            # utils_hist.hist_tensor2D(x_ch = prediction, dir='score', name='score', en=True, log=True)
        else:
            prediction[prediction < 0.0] = 0.0

            if learn_scaled_values:
                img_interest_pt_cur = prediction.astype(np.uint16)
                score_th = 127
            else:
                img_interest_pt_cur = prediction
                score_th = 0.001

            # at boundary pixles score/desc will be wrong so do not write pred qty near the borader.
            guard_band = 32 if args.write_desc_type == 'PRED' else 0

            write_desc.write_score_desc_as_text(desc_tensor_cur=desc_tensor, img_interest_pt_cur=img_interest_pt_cur,
                                                txt_file_name=txt_file_name, score_th=score_th,
                                                skip_fac_for_reading_desc=1, en_nms=args.en_nms,
                                                scale_to_write_kp_loc_to_orig_res=scale_to_write_kp_loc_to_orig_res,
                                                recursive_nms=True, learn_scaled_values=learn_scaled_values,
                                                guard_band=guard_band, true_nms=True, f2r=args.do_pred_cordi_f2r)


      #utils_hist.hist_tensor2D(x_ch = prediction, dir='score', name='score', en=True, log=True)
    else:  
      prediction[prediction < 0.0] = 0.0
      
      if learn_scaled_values:
        img_interest_pt_cur = prediction.astype(np.uint16)
        score_th = 127
      else:  
        img_interest_pt_cur = prediction
        score_th = 0.001

      # at boundary pixles score/desc will be wrong so do not write pred qty near the borader.
      guard_band = 32 if args.write_desc_type == 'PRED' else 0

      write_desc.write_score_desc_as_text(desc_tensor_cur = desc_tensor, img_interest_pt_cur = img_interest_pt_cur,
        txt_file_name = txt_file_name, score_th = score_th, skip_fac_for_reading_desc = 1, en_nms=args.en_nms,
        scale_to_write_kp_loc_to_orig_res = scale_to_write_kp_loc_to_orig_res,
        recursive_nms=True, learn_scaled_values=learn_scaled_values, guard_band = guard_band, true_nms=True, f2r=args.do_pred_cordi_f2r)

def viz_depth(prediction = [], args=[], output_name=[], input_name=[]):
    max_value_depth = args.max_depth
    output_image = torch.tensor(prediction)
    if args.viz_depth_color_type == 'rainbow':
        not_valid_indices = output_image == 0
        output_image = 255*xnn.utils.tensor2array(output_image, max_value=max_value_depth, colormap='rainbow')[:,:,::-1]
        output_image[not_valid_indices] = 0
    elif args.viz_depth_color_type == 'rainbow_blend':
        print(max_value_depth)
        #scale_mul = 1 if args.visualize_gt else 255
        print(output_image.min())
        print(output_image.max())
        not_valid_indices = output_image == 0
        output_image = 255*xnn.utils.tensor2array(output_image, max_value=max_value_depth, colormap='rainbow')[:,:,::-1]
        print(output_image.max())
        #output_image[label == 1] = 0
        input_bgr = cv2.imread(input_name)  # Read the actual RGB image
        input_bgr = cv2.resize(input_bgr, dsize=(prediction.shape[1], prediction.shape[0]))
        if args.sky_dir:
            label_file = os.path.join(args.sky_dir, seq, seq + '_image_00_' + base_file)
            label = cv2.imread(label_file)
            label = cv2.resize(label, dsize=(prediction.shape[1], prediction.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            output_image[label == 1] = 0
        output_image[not_valid_indices] = 0
        output_image = xnn.utils.chroma_blend(input_bgr, output_image)  # chroma_blend(input_bgr, output_image)

    elif args.viz_depth_color_type == 'bone':
        output_image = 255 * xnn.utils.tensor2array(output_image, max_value=max_value_depth, colormap='bone')
    elif args.viz_depth_color_type == 'raw_depth':
        output_image = np.array(output_image)
        output_image[output_image > max_value_depth] = max_value_depth
        output_image[output_image < 0] = 0
        scale = 2.0**16 - 1.0 #255
        output_image = (output_image / max_value_depth) * scale
        output_image = output_image.astype(np.uint16)
        # output_image[(label[:,:,0]==1)|(label[:,:,0]==4)]=255
    elif args.viz_depth_color_type == 'plasma':
        plt.imsave(output_name, output_image, cmap='plasma', vmin=0, vmax=max_value_depth)
    elif args.viz_depth_color_type == 'log_greys':        
        plt.imsave(output_name, np.log10(output_image), cmap='Greys', vmin=0, vmax=np.log10(max_value_depth))
        #plt.imsave(output_name, output_image, cmap='Greys', vmin=0, vmax=max_value_depth)
    else:
        print("undefined color type for visualization")
        exit(0)

    if args.viz_depth_color_type != 'plasma':
        # plasma type will be handled by imsave
        cv2.imwrite(output_name, output_image)


def wrapper_write_desc(args=[], task_index=0, outputs=[], index=0, output_name=[], output_name_short=[]):
    if args.write_desc_type == 'GT':
        # write GT desc
        tensor_to_write = target_list[task_index]
    elif args.write_desc_type == 'PRED':
        # write predicted desc
        tensor_to_write = outputs[task_index]

    interest_pt_score = np.array(tensor_to_write[index, 0, ...])

    if args.make_score_zero_mean:
        # visulization code assumes range [0,255]. Add 128 to make range the same in case of zero mean too.
        interest_pt_score += 128.0

    if args.write_desc_type == 'NONE':
        # scale + clip score between 0-255 and convert score_array to image
        # scale_range = 127.0/0.005
        # scale_range = 255.0/np.max(interest_pt_score)
        scale_range = 1.0
        interest_pt_score = np.clip(interest_pt_score * scale_range, 0.0, 255.0)
        interest_pt_score = np.asarray(interest_pt_score, 'uint8')

    interest_pt_descriptor = np.array(tensor_to_write[index, 1:, ...])

    # output_name = os.path.join(infer_path[task_index], seq + '_' + input_path[-1][index].split('/')[-3] + '_' + base_file)
    cv2.imwrite(output_name, interest_pt_score)

    # output_name_short = os.path.join(infer_path[task_index], os.path.basename(input_path[-1][index]))

    scale_to_write_kp_loc_to_orig_res = args.scale_to_write_kp_loc_to_orig_res
    if args.scale_to_write_kp_loc_to_orig_res[0] == -1:
        scale_to_write_kp_loc_to_orig_res[0] = input_list[task_index].shape[2] / target_list[task_index].shape[2]
        scale_to_write_kp_loc_to_orig_res[1] = scale_to_write_kp_loc_to_orig_res[0]

    print("scale_to_write_kp_loc_to_orig_res: ", scale_to_write_kp_loc_to_orig_res)
    store_desc(args=args, output_name=output_name_short, desc_tensor=interest_pt_descriptor,
               prediction=interest_pt_score,
               scale_to_write_kp_loc_to_orig_res=scale_to_write_kp_loc_to_orig_res,
               learn_scaled_values=args.learn_scaled_values_interest_pt,
               write_dense=False)


def get_transforms(args):
    # image normalization can be at the beginning of transforms or at the end
    args.image_mean = np.array(args.image_mean, dtype=np.float32)
    args.image_scale = np.array(args.image_scale, dtype=np.float32)
    image_prenorm = xvision.transforms.image_transforms.NormalizeMeanScale(mean=args.image_mean, scale=args.image_scale) if args.image_prenorm else None
    image_postnorm = xvision.transforms.image_transforms.NormalizeMeanScale(mean=args.image_mean, scale=args.image_scale) if (not image_prenorm) else None

    #target size must be according to output_size. prediction will be resized to output_size before evaluation.
    test_transform = xvision.transforms.image_transforms.Compose([
        image_prenorm,
        xvision.transforms.image_transforms.AlignImages(),
        xvision.transforms.image_transforms.MaskTarget(args.target_mask, 0),
        xvision.transforms.image_transforms.CropRect(args.img_border_crop),
        xvision.transforms.image_transforms.Scale(args.img_resize, target_size=args.output_size, is_flow=args.is_flow),
        image_postnorm,
        xvision.transforms.image_transforms.ConvertToTensor()
        ])

    return test_transform


def _upsample_impl(tensor, output_size, upsample_mode):
    # upsample of long tensor is not supported currently. covert to float, just to avoid error.
    # we can do thsi only in the case of nearest mode, otherwise output will have invalid values.
    convert_to_float = False
    if isinstance(tensor, (torch.LongTensor,torch.cuda.LongTensor)):
        convert_to_float = True
        tensor = tensor.float()
        upsample_mode = 'nearest'
    #

    dim_added = False
    if len(tensor.shape) < 4:
        tensor = tensor[np.newaxis,...]
        dim_added = True
    #
    if (tensor.size()[-2:] != output_size):
        tensor = torch.nn.functional.interpolate(tensor, output_size, mode=upsample_mode)
    # --
    if dim_added:
        tensor = tensor[0,...]
    #

    if convert_to_float:
        tensor = tensor.long()
    #
    return tensor

def upsample_tensors(tensors, output_sizes, upsample_mode):
    if not output_sizes:
        return tensors
    #
    if isinstance(tensors, (list,tuple)):
        for tidx, tensor in enumerate(tensors):
            tensors[tidx] = _upsample_impl(tensor, output_sizes[tidx][-2:], upsample_mode)
        #
    else:
        tensors = _upsample_impl(tensors, output_sizes[0][-2:], upsample_mode)
    return tensors




def eval_output(args, output, label, confusion_matrix, n_classes):
    if len(label.shape)>2:
        label = label[:,:,0]
    gt_labels = label.ravel()
    det_labels = output.ravel().clip(0,n_classes)
    gt_labels_valid_ind = np.where(gt_labels != 255)
    gt_labels_valid = gt_labels[gt_labels_valid_ind]
    det_labels_valid = det_labels[gt_labels_valid_ind]
    for r in range(confusion_matrix.shape[0]):
        for c in range(confusion_matrix.shape[1]):
            confusion_matrix[r,c] += np.sum((gt_labels_valid==r) & (det_labels_valid==c))

    return confusion_matrix
    
def compute_accuracy(args, confusion_matrix, n_classes):
    num_selected_classes = n_classes
    tp = np.zeros(n_classes)
    population = np.zeros(n_classes)
    det = np.zeros(n_classes)
    iou = np.zeros(n_classes)
    
    for r in range(n_classes):
      for c in range(n_classes):
        population[r] += confusion_matrix[r][c]
        det[c] += confusion_matrix[r][c]   
        if r == c:
          tp[r] += confusion_matrix[r][c]

    for cls in range(num_selected_classes):
      intersection = tp[cls]
      union = population[cls] + det[cls] - tp[cls]
      iou[cls] = (intersection / union) if union else 0     # For caffe jacinto script
      #iou[cls] = (intersection / (union + np.finfo(np.float32).eps))  # For pytorch-jacinto script

    num_nonempty_classes = 0
    for pop in population:
      if pop>0:
        num_nonempty_classes += 1
          
    mean_iou = np.sum(iou) / num_nonempty_classes if num_nonempty_classes else 0
    accuracy = np.sum(tp) / np.sum(population) if np.sum(population) else 0
    
    #F1 score calculation
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1_score = np.zeros(n_classes)

    for cls in range(num_selected_classes):
        fp[cls] = det[cls] - tp[cls]
        fn[cls] = population[cls] - tp[cls]
        precision[cls] = tp[cls] / (det[cls] + 1e-10)
        recall[cls] = tp[cls] / (tp[cls] + fn[cls] + 1e-10)        
        f1_score[cls] = 2 * precision[cls]*recall[cls] / (precision[cls] + recall[cls] + 1e-10)

    return accuracy, mean_iou, iou, f1_score
    
        
def infer_video(args, net):
    videoIpHandle = imageio.get_reader(args.input, 'ffmpeg')
    fps = math.ceil(videoIpHandle.get_meta_data()['fps'])
    print(videoIpHandle.get_meta_data())
    numFrames = min(len(videoIpHandle), args.num_images)
    videoOpHandle = imageio.get_writer(args.output,'ffmpeg', fps=fps)
    for num in range(numFrames):
        print(num, end=' ')
        sys.stdout.flush()
        input_blob = videoIpHandle.get_data(num)
        input_blob = input_blob[...,::-1]    #RGB->BGR
        output_blob = infer_blob(args, net, input_blob)     
        output_blob = output_blob[...,::-1]  #BGR->RGB            
        videoOpHandle.append_data(output_blob)
    videoOpHandle.close()
    return


def absreldiff(x, y, eps = 0.0, max_val=None):
    assert x.size() == y.size(), 'tensor dimension mismatch'
    if max_val is not None:
        x = torch.clamp(x, -max_val, max_val)
        y = torch.clamp(y, -max_val, max_val)
    #

    diff = torch.abs(x - y)
    y = torch.abs(y)

    den_valid = (y == 0).float()
    eps_arr = (den_valid * (1e-6))   # Just to avoid divide by zero

    large_arr = (y > eps).float()    # ARD is not a good measure for small ref values. Avoid them.
    out = (diff / (y + eps_arr)) * large_arr
    return out


def absreldiff_rng3to80(x, y):
    return absreldiff(x, y, eps = 3.0, max_val=80.0)



def create_video(args, infer_path):
    op_file_name = args.data_path.split('/')[-1]
    os.system(' ffmpeg -framerate 30 -pattern_type glob -i "{}/*.png" -c:v libx264 -vb 50000k -qmax 20 -r 10 -vf scale=1024:512  -pix_fmt yuv420p {}.MP4'.format(infer_path, op_file_name))

def write_onnx_model(args, model, save_path, name='checkpoint.onnx'):
    is_cuda = next(model.parameters()).is_cuda
    input_list = create_rand_inputs(args, is_cuda=is_cuda)
    #
    model.eval()
    torch.onnx.export(get_model_orig(model), input_list, os.path.join(save_path, name), export_params=True, verbose=False,
                      do_constant_folding=True, opset_version=args.opset_version)
    # torch onnx export does not update names. Do it using onnx.save



if __name__ == '__main__':
    train_args = get_config()
    main(train_args)
