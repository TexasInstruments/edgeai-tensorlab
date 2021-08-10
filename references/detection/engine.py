import os
import math
import sys
import time
import torch
from colorama import Fore

import torchvision.models.detection.mask_rcnn
from torchvision import xnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import torchinfo
import utils


def train_one_epoch(args, model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # warmup was earlier here, but is now included in the original lr_scheduler
    # see creating of lr_scheduler in train.py

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(args, model, data_loader, device, synchronize_time=False):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if synchronize_time and torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): detach_dict(output) for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


def complexity(args, model):
    model.eval()
    image_size_tuple = args.input_size if isinstance(args.input_size, (list,tuple)) else \
        (args.input_size, args.input_size)
    data_shape = (1,3,*image_size_tuple)
    torchinfo.summary(model, data_shape, depth=10)


def detach_dict(d):
    for k in d:
        d[k].detach()
    #
    return d


def export(args, model, model_name=None):
    model.eval()
    model_name = 'model' if model_name is None else model_name
    image_size_tuple = args.input_size if isinstance(args.input_size, (list,tuple)) else \
        (args.input_size, args.input_size)
    data_shape = (1,3,*image_size_tuple)
    example_input = torch.rand(*data_shape)
    torch.onnx.export(model, example_input, os.path.join(args.output_dir, model_name+'.onnx'), opset_version=args.opset_version)
    #script_model = torch.jit.trace(model, example_input, strict=False)
    #torch.jit.save(script_model, os.path.join(args.output_dir, model_name+'_model.pth'))

