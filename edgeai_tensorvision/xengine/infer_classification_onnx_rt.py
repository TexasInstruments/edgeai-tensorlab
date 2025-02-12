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
import sys
import shutil
import time
import datetime

import random
import numpy as np
from colorama import Fore
import random
import progiter
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import onnx
import onnxruntime

from edgeai_torchmodelopt import xnn
from edgeai_tensorvision import xvision


# ################################################
def get_config():
    args = xnn.utils.ConfigNode()
    args.model_config = xnn.utils.ConfigNode()
    args.dataset_config = xnn.utils.ConfigNode()

    args.model_name = 'mobilenet_v2_classification'     # model architecture'
    args.dataset_name = 'imagenet_classification'       # image folder classification

    args.data_path = './data/datasets/ilsvrc'           # path to dataset
    args.save_path = None                               # checkpoints save path
    args.pretrained = './data/modelzoo/pretrained/pytorch/imagenet_classification/ericsun99/MobileNet-V2-Pytorch/mobilenetv2_Top1_71.806_Top2_90.410.pth.tar' # path to pre_trained model

    args.workers = 8                                    # number of data loading workers (default: 4)
    args.batch_size = 256                               # mini_batch size (default: 256)
    args.print_freq = 100                               # print frequency (default: 100)

    args.img_resize = 256                               # image resize
    args.img_crop = 224                                 # image crop

    args.image_mean = (123.675, 116.28, 103.53)         # image mean for input image normalization')
    args.image_scale = (0.017125, 0.017507, 0.017429)   # image scaling/mult for input iamge normalization')

    args.logger = None                                  # logger stream to output into

    args.data_augument = 'inception'                    # data augumentation method, choices=['inception','resize','adaptive_resize']
    args.dataset_format = 'folder'                      # dataset format, choices=['folder','lmdb']
    args.count_flops = True                             # count flops and report

    args.lr_calib = 0.05                                # lr for bias calibration

    args.rand_seed = 1                                  # random seed
    args.save_onnx = False                          # apply quantized inference or not
    args.print_model = False                            # print the model to text
    args.run_soon = True                                # Set to false if only cfs files/onnx  modelsneeded but no training
    args.parallel_model = True                          # parallel or not
    args.shuffle = True                                 # shuffle or not
    args.epoch_size = 0                                 # epoch size
    args.rand_seed = 1                                  # random seed
    args.date = None                                    # date to add to save path. if this is None, current date will be added.
    args.write_layer_ip_op = False
    args.gpu_mode = True                                # False will make inference run on CPU

    args.quantize = False                               # apply quantized inference or not
    #args.model_surgery = None                           # replace activations with PAct2 activation module. Helpful in quantized training.
    args.bitwidth_weights = 8                           # bitwidth for weights
    args.bitwidth_activations = 8                       # bitwidth for activations
    args.histogram_range = True                         # histogram range for calibration
    args.per_channel_q = False                          # apply separate quantizion factor for each channel in depthwise or not
    args.bias_calibration = False                        # apply bias correction during quantized inference calibration

    args.opset_version = 9                              # onnx opset version
    return args


def main(args):
    assert not hasattr(args, 'model_surgery'), 'the argument model_surgery is deprecated, it is not needed now - remove it'

    if (args.phase == 'validation' and args.bias_calibration):
        args.bias_calibration = False
        warnings.warn('switching off bias calibration in validation')
    #

    if args.save_path is None:
        save_path = get_save_path(args)
    else:
        save_path = args.save_path
    #

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #

    #################################################
    if args.logger is None:
        log_file = os.path.splitext(os.path.basename(__file__))[0] + '.log'
        args.logger = xnn.utils.TeeLogger(filename=os.path.join(save_path,log_file))

    #################################################
    # global settings. rand seeds for repeatability
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    ################################
    # print everything for log
    # reset character color, in case it is different
    print('{}'.format(Fore.RESET))
    print("=> args: ", args)
    print("=> resize resolution: {}".format(args.img_resize))
    print("=> crop resolution  : {}".format(args.img_crop))
    sys.stdout.flush()


    #################################################
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    model = onnx.load(args.pretrained)
    # Run the ONNX model with Caffe2
    onnx.checker.check_model(model)

    val_loader = get_data_loaders(args)
    validate(args, val_loader, model, criterion)

    if args.logger is not None:
        args.logger.close()
        args.logger = None


def validate(args, val_loader, model, criterion):

    # change color to green
    print('{}'.format(Fore.GREEN), end='')

    session = onnxruntime.InferenceSession(args.pretrained, None)
    input_name = session.get_inputs()[0].name
    input_details = session.get_inputs()
    output_details = session.get_outputs()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    use_progressbar = True
    epoch_size = get_epoch_size(args, val_loader, args.epoch_size)

    if use_progressbar:
        progress_bar = progiter.ProgIter(np.arange(epoch_size), chunksize=1)
        last_update_iter = -1

    end = time.time()
    for iteration, (input, target) in enumerate(val_loader):
        if args.gpu_mode:
            input_list = [img.cuda() for img in input_list]
            target = target.cuda(non_blocking=True)
            input = torch.cat([j.cuda() for j in input], dim=1) if (type(input) in (list,tuple)) else input.cuda()
        # compute output
        output = session.run([], {input_name: np.asarray(input)})
        output = [torch.tensor(output[index]) for index in range(len(output))]
        if type(output) in (list, tuple):
            output = output[0]
        #

        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        final_iter = (iteration >= (epoch_size-1))

        if ((iteration % args.print_freq) == 0) or final_iter:
            status_str = 'Time {batch_time.val:.2f}({batch_time.avg:.2f}) LR {cur_lr:.4f} ' \
                         'Loss {loss.val:.2f}({loss.avg:.2f}) Prec@1 {top1.val:.2f}({top1.avg:.2f}) Prec@5 {top5.val:.2f}({top5.avg:.2f})' \
                         .format(batch_time=batch_time, cur_lr=0.0, loss=losses, top1=top1, top5=top5)
            #
            prefix = '**' if final_iter else '=>'
            if use_progressbar:
                progress_bar.set_description('{} validation'.format(prefix))
                progress_bar.set_postfix(dict(Epoch='{}'.format(status_str)))
                progress_bar.update(iteration - last_update_iter)
                last_update_iter = iteration
            else:
                iter_str = '{:6}/{:6}    : '.format(iteration+1, len(val_loader))
                status_str = prefix + ' ' + iter_str + status_str
                if final_iter:
                    xnn.utils.print_color(status_str, color=Fore.GREEN)
                else:
                    xnn.utils.print_color(status_str)

        if final_iter:
            break

        if use_progressbar:
            progress_bar.close()

        # to print a new line - do not provide end=''
        print('{}'.format(Fore.RESET), end='')

    return top1.avg


#######################################################################
def get_save_path(args, phase=None):
    date = args.date if args.date else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path_base = os.path.join('./data/checkpoints/edgeailite', args.dataset_name, date + '_' + args.dataset_name + '_' + args.model_name)
    save_path = save_path_base + '_resize{}_crop{}'.format(args.img_resize, args.img_crop)
    phase = phase if (phase is not None) else args.phase
    save_path = os.path.join(save_path, phase)
    return save_path


def get_model_orig(model):
    is_parallel_model = isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    model_orig = (model.module if is_parallel_model else model)
    model_orig = (model_orig.module if isinstance(model_orig, (edgeai_torchmodelopt.xmodelopt.quantization.v1.QuantBaseModule)) else model_orig)
    return model_orig


def create_rand_inputs(args, is_cuda=True):
    x = torch.rand((1, args.model_config.input_channels, args.img_crop, args.img_crop))
    x = x.cuda() if is_cuda else x
    return x


def count_flops(args, model):
    is_cuda = next(model.parameters()).is_cuda
    input_list = create_rand_inputs(args, is_cuda)
    model.eval()
    total_mult_adds, total_params = xnn.utils.get_model_complexity(model, input_list)
    total_mult_adds_giga = total_mult_adds/1e9
    total_flops = total_mult_adds_giga*2
    total_params_mega = total_params/1e6
    print('=> Resize = {}, Crop = {}, GFLOPs = {}, GMACs = {}, MegaParams = {}'.format(args.img_resize, args.img_crop, total_flops, total_mult_adds_giga, total_params_mega))


def write_onnx_model(args, model, save_path, name='checkpoint.onnx'):
    is_cuda = next(model.parameters()).is_cuda
    dummy_input = create_rand_inputs(args, is_cuda)
    #
    model.eval()
    torch.onnx.export(model, dummy_input, os.path.join(save_path,name), export_params=True, verbose=False,
                      do_constant_folding=True, opset_version=args.opset_version)

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_epoch_size(args, loader, args_epoch_size):
    if args_epoch_size == 0:
        epoch_size = len(loader)
    elif args_epoch_size < 1:
        epoch_size = int(len(loader) * args_epoch_size)
    else:
        epoch_size = min(len(loader), int(args_epoch_size))
    return epoch_size


def get_data_loaders(args):
    # Data loading code
    normalize = xvision.transforms.NormalizeMeanScale(mean=args.image_mean, scale=args.image_scale) \
                        if (args.image_mean is not None and args.image_scale is not None) else None

    # pass tuple to Resize() to resize to exact size without respecting aspect ratio (typical caffe style)
    val_transform = xvision.transforms.Compose([xvision.transforms.Resize(size=args.img_resize),
                                               xvision.transforms.CenterCrop(size=args.img_crop),
                                               xvision.transforms.ToFloat(),
                                               xvision.transforms.ToTensor(),
                                               normalize])

    val_dataset = xvision.datasets.classification.__dict__[args.dataset_name](args.dataset_config, args.data_path, transforms=val_transform)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.workers,
                                             pin_memory=True, drop_last=False)

    return val_loader


#################################################
def shape_as_string(shape=[]):
    shape_str = ''
    for dim in shape:
        shape_str += '_' + str(dim)
    return shape_str


def write_tensor_int(m=[], tensor=[], suffix='op', bitwidth=8, power2_scaling=True, file_format='bin',
                     rnd_type='rnd_sym'):
    mn = tensor.min()
    mx = tensor.max()

    print(
        '{:6}, {:32}, {:10}, {:7.2f}, {:7.2f}'.format(suffix, m.name, m.__class__.__name__, tensor.min(), tensor.max()),
        end=" ")

    [tensor_scale, clamp_limits] = xnn.utils.compute_tensor_scale(tensor, mn, mx, bitwidth, power2_scaling)
    print("{:30} : {:15} : {:8.2f}".format(str(tensor.shape), str(tensor.dtype), tensor_scale), end=" ")

    print_weight_bias = False
    if rnd_type == 'rnd_sym':
        # use best rounding for offline quantities
        if suffix == 'weight' and print_weight_bias:
            no_idx = 0
            torch.set_printoptions(precision=32)
            print("tensor_scale: ", tensor_scale)
            print(tensor[no_idx])
        if tensor.dtype != torch.int64:
            tensor = xnn.utils.symmetric_round_tensor(tensor * tensor_scale)
        if suffix == 'weight' and print_weight_bias:
            print(tensor[no_idx])
    else:
        # for activation use HW friendly rounding
        if tensor.dtype != torch.int64:
            tensor = xnn.utils.upward_round_tensor(tensor * tensor_scale)
    tensor = tensor.clamp(clamp_limits[0], clamp_limits[1]).float()

    if bitwidth == 8:
        data_type = np.int8
    elif bitwidth == 16:
        data_type = np.int16
    elif bitwidth == 32:
        data_type = np.int32
    else:
        exit("Bit width other 8,16,32 not supported for writing layer level op")

    tensor = tensor.cpu().numpy().astype(data_type)

    print("{:7} : {:7d} : {:7d}".format(str(tensor.dtype), tensor.min(), tensor.max()))

    tensor_dir = './data/checkpoints/edgeailite/debug/test_vecs/' + '{}_{}_{}_scale_{:010.4f}'.format(m.name,
                                                                                            m.__class__.__name__,
                                                                                            suffix, tensor_scale)

    if not os.path.exists(tensor_dir):
        os.makedirs(tensor_dir)

    if file_format == 'bin':
        tensor_name = tensor_dir + "/{}_shape{}.bin".format(m.name, shape_as_string(shape=tensor.shape))
        tensor.tofile(tensor_name)
    elif file_format == 'npy':
        tensor_name = tensor_dir + "/{}_shape{}.npy".format(m.name, shape_as_string(shape=tensor.shape))
        np.save(tensor_name, tensor)
    else:
        warnings.warn('unknown file_format for write_tensor - no file written')
    #

    # utils_hist.comp_hist_tensor3d(x=tensor, name=m.name, en=True, dir=m.name, log=True, ch_dim=0)


def write_tensor_float(m=[], tensor=[], suffix='op'):
    mn = tensor.min()
    mx = tensor.max()

    print(
        '{:6}, {:32}, {:10}, {:7.2f}, {:7.2f}'.format(suffix, m.name, m.__class__.__name__, tensor.min(), tensor.max()))
    root = os.getcwd()
    tensor_dir = root + '/checkpoints/debug/test_vecs/' + '{}_{}_{}'.format(m.name, m.__class__.__name__, suffix)

    if not os.path.exists(tensor_dir):
        os.makedirs(tensor_dir)

    tensor_name = tensor_dir + "/{}_shape{}.npy".format(m.name, shape_as_string(shape=tensor.shape))
    np.save(tensor_name, tensor.data)


def write_tensor(data_type='int', m=[], tensor=[], suffix='op', bitwidth=8, power2_scaling=True, file_format='bin',
                 rnd_type='rnd_sym'):
    if data_type == 'int':
        write_tensor_int(m=m, tensor=tensor, suffix=suffix, rnd_type=rnd_type, file_format=file_format)
    elif data_type == 'float':
        write_tensor_float(m=m, tensor=tensor, suffix=suffix)


enable_hook_function = True
def write_tensor_hook_function(m, inp, out, file_format='bin'):
    if not enable_hook_function:
        return

    #Output
    if isinstance(out, (torch.Tensor)):
        write_tensor(m=m, tensor=out, suffix='op', rnd_type ='rnd_up', file_format=file_format)

    #Input(s)
    if type(inp) is tuple:
        #if there are more than 1 inputs
        for index, sub_ip in enumerate(inp[0]):
            if isinstance(sub_ip, (torch.Tensor)):
                write_tensor(m=m, tensor=sub_ip, suffix='ip_{}'.format(index), rnd_type ='rnd_up', file_format=file_format)
    elif isinstance(inp, (torch.Tensor)):
         write_tensor(m=m, tensor=inp, suffix='ip', rnd_type ='rnd_up', file_format=file_format)

    #weights
    if hasattr(m, 'weight'):
        if isinstance(m.weight,torch.Tensor):
            write_tensor(m=m, tensor=m.weight, suffix='weight', rnd_type ='rnd_sym', file_format=file_format)

    #bias
    if hasattr(m, 'bias'):
        if m.bias is not None:
            write_tensor(m=m, tensor=m.bias, suffix='bias', rnd_type ='rnd_sym', file_format=file_format)


if __name__ == '__main__':
    main()
