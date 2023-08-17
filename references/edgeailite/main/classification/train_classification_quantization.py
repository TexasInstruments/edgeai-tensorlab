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

#####################################################################################

# Post Training Quantization (PTQ) / Quantization Aware Training (QAT) Example
# this original code is from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
# the changes required for quantizing the model are under the flag args.quantize
#
# BSD 3-Clause License
#
# Copyright (c) 2017,
# All rights reserved.
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

import argparse
import os
import random
import shutil
import time
import warnings
import onnx

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# some of the default torchvision models need some minor tweaks to be friendly for
# quantization aware training. so use models from references.edgeailite insead
#import torchvision.models as models

from torchvision.edgeailite import xnn
from torchvision.edgeailite import xvision
from torchvision.edgeailite.xvision import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def str2bool(v):
  if isinstance(v, (str)):
      if v.lower() in ("yes", "true", "t", "1"):
          return True
      elif v.lower() in ("no", "false", "f", "0"):
          return False
      else:
          return v
      #
  else:
      return v

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='mobilenetv2_x1',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: mobilenetv2_x1)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_size', default=0, type=float, metavar='N',
                    help='fraction of training epoch to use. 0 (default) means full training epoch')
parser.add_argument('--epoch_size_val', default=0, type=float, metavar='N',
                    help='fraction of validation epoch to use. 0 (default) means full validation epoch')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', default=30, type=int,
                    metavar='N', help='number of steps before learning rate is reduced')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', type=str, default=None,
                    help='use pre-trained model')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--use_gpu', default=True, type=str2bool,
                    help='whether to use gpu or not')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--save_path', type=str, default='./data/checkpoints/quantization',
                    help='path to save the logs and models')
parser.add_argument('--quantize', default=True, type=bool,
                    help='Enable Quantization')
parser.add_argument('--opset_version', default=11, type=int,
                    help='opset version for onnx export')
parser.add_argument('--img_resize', type=int, default=256,
                    help='images will be first resized to this size during training and validation')
parser.add_argument('--img_crop', type=int, default=224,
                    help='the cropped portion (validation), cropped pertion will be resized to this size (training)')

best_acc1 = 0


def main():
    args = parser.parse_args()

    args.cur_lr = args.lr

    if args.use_gpu is None:
        args.use_gpu = (not args.quantize)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.quantize:
        assert args.pretrained is not None and args.pretrained is not False, \
            'pretrained checkpoint must be provided for QAT'

    dummy_input = torch.rand((1, 3, args.img_crop, args.img_crop))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    model, change_names_dict = model if isinstance(model, (list,tuple)) else (model, None)

    if args.quantize:
        model = xnn.quantization.QuantTrainModule(model, dummy_input=dummy_input)
    #

    if args.pretrained is not None:
        print("=> using pre-trained model for {} from {}".format(args.arch, args.pretrained))
	    # for QAT, the original one is in model.module
        model_orig = model.module if args.quantize else model
        if hasattr(model_orig, 'load_weights'):
            model_orig.load_weights(args.pretrained, download_root='./data/downloads', change_names_dict=change_names_dict)
        else:
            xnn.utils.load_weights(model_orig, args.pretrained, download_root='./data/downloads', change_names_dict=change_names_dict)
        #		

    if args.use_gpu:
        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.NormalizeMeanScale(mean=[123.675, 116.28, 103.53], scale=[0.017125, 0.017507, 0.017429])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(args.img_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToFloat(),   # converting to float avoids the division by 255 in ToTensor()
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(args.img_resize),
        transforms.CenterCrop(args.img_crop),
        transforms.ToFloat(),  # converting to float avoids the division by 255 in ToTensor()
        transforms.ToTensor(),
        normalize,
    ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = get_dataset_sampler(train_dataset, args.epoch_size) if args.epoch_size != 0 else None
        val_sampler = get_dataset_sampler(val_dataset, args.epoch_size_val) if args.epoch_size_val != 0 else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    ##validate(val_loader, model, criterion, args)

    if args.evaluate:
        return

    for epoch in range(args.start_epoch, args.epochs):
        model_orig = model.module if isinstance(model,
            (torch.nn.parallel.DistributedDataParallel, torch.nn.parallel.DataParallel)) else model

        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # distill mode needs learning rate inside the model.
        if hasattr(model_orig, 'set_learning_rate') and callable(model_orig.set_learning_rate):
            model_orig.set_learning_rate(learning_rate=args.cur_lr)
        #

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        model_orig = model_orig.module if args.quantize else model_orig
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            out_basename = args.arch + ('_checkpoint_quantized.pth' if args.quantize else '_checkpoint.pth')
            save_filename = os.path.join(args.save_path, out_basename)
            checkpoint_dict = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model_orig.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': (optimizer.state_dict() if optimizer is not None else None),
            }
            save_checkpoint(checkpoint_dict, is_best, filename=save_filename)
            # onnx model export
            save_onnxname = os.path.splitext(save_filename)[0]+'.onnx'
            write_onnx_model(args, model_orig, is_best, filename=save_onnxname)


    if args.quantize and hasattr(model, 'convert') and callable(model.convert):
        model_quant.eval()
        model_quant.cpu()
        model_quant.convert()
        if hasattr(model_quant, 'export') and callable(model_quant.export):
            model_quant = model_quant.export(dummy_input)
        else:
            model_quant = torch.jit.trace(model_quant, dummy_input)
        #

        print('########################################')
        print(model_quant)

        print('########################################')
        print(model_quant.graph)

        print('########################################')
        print('saving the quantized model')
        out_basename = args.arch + ('_model_quantized.pth' if args.quantize else '_model.pth')
        save_filename = os.path.join(args.save_path, out_basename)
        model_quant.save(save_filename)

        print('########################################')
        # evaluate on validation set
        acc1 = validate(val_loader, model_quant, criterion, args)
        print('########################################')


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    # freeze the quantization params and bn
    if args.quantize:
        if epoch > 2 and epoch > ((args.epochs//2)-1):
            xnn.utils.print_once('Freezing BN for subseq epochs')
            # model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            # this is a more generic version
            xnn.utils.freeze_bn(model)
        #
        if epoch > 4 and epoch >= ((args.epochs//2)+1):
            xnn.utils.print_once('Freezing range observer for subseq epochs')
            # model.apply(torch.quantization.disable_observer)
            # this is a more generic version
            xnn.layers.freeze_quant_range(model)
        #
    #

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_gpu:
            # GPU/CUDA is not yet support for Torch quantization
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        #

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, args.cur_lr)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.use_gpu:
                # GPU/CUDA is not yet support for Torch quantization
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            #

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i, args.cur_lr)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    dirname = os.path.dirname(filename)
    xnn.utils.makedir_exist_ok(dirname)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.splitext(filename)[0]+'_best.pth')


def create_rand_inputs(is_cuda):
    dummy_input = torch.rand((1, 3, args.img_crop, args.img_crop))
    dummy_input = dummy_input.cuda() if is_cuda else dummy_input
    return dummy_input


def write_onnx_model(args, model, is_best, filename='checkpoint.onnx'):
    model.eval()
    is_cuda = next(model.parameters()).is_cuda
    dummy_input = create_rand_inputs(is_cuda)
    torch.onnx.export(model, dummy_input, filename, export_params=True, verbose=False,
                      do_constant_folding=True, opset_version=args.opset_version)
    onnx.shape_inference.infer_shapes_path(filename, filename)
    if is_best:
        shutil.copyfile(filename, os.path.splitext(filename)[0]+'_best.onnx')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.lr_fmtstr = self._get_lr_fmtstr()
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, cur_lr):
        entries = [self.prefix + self.batch_fmtstr.format(batch), self.lr_fmtstr.format(cur_lr)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def _get_lr_fmtstr(self):
        fmt = 'LR {:g}'
        return fmt

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_step_size))
    args.cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_dataset_sampler(dataset_object, epoch_size):
    num_samples = len(dataset_object)
    epoch_size = num_samples if (epoch_size == 0) else \
        (int(epoch_size * num_samples) if epoch_size < 1.0 else int(epoch_size))
    print('=> creating a random sampler as epoch_size is specified')
    dataset_sampler = torch.utils.data.sampler.RandomSampler(data_source=dataset_object, replacement=True, num_samples=epoch_size)
    return dataset_sampler


if __name__ == '__main__':
    main()