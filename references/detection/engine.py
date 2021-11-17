import os
import math
import random
import sys
import time
import warnings
import onnx
import torch
from colorama import Fore
import collections
import torchinfo

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator

import utils
import export_proto

from torchvision.edgeailite import xnn


def train_one_epoch(args, model, optimizer, data_loader, device, epoch, print_freq, summary_writer=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = min(print_freq, len(data_loader))

    # warmup was earlier here, but is now included in the original lr_scheduler
    # see creating of lr_scheduler in train.py

    for batch_id, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
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

        if hasattr(args, 'quit_event') and args.quit_event.is_set():
            break

    if summary_writer:
        summary_writer.add_scalar('train/loss @ epoch', metric_logger.meters['loss'].avg, epoch)

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
def evaluate(args, model, data_loader, device, epoch, synchronize_time=False, print_freq=100, summary_writer=None):
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
    saved_images = collections.deque(maxlen=2)
    saved_results = collections.deque(maxlen=2)
    batch_counter = 0
    print_freq = min(print_freq, len(data_loader))

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(img.to(device) for img in images)

        if synchronize_time and torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        if saved_results is None or random.random() < (1.0/(batch_counter+1)):
            saved_images.append(images[0])
            saved_results.append(res)

        if hasattr(args, 'quit_event') and args.quit_event.is_set():
            break

        batch_counter += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    if hasattr(args, 'quit_event') and args.quit_event.is_set():
        return

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    if summary_writer:
        # save accuracy value
        # AP
        accuracy_mean_ap = coco_evaluator.coco_eval['bbox'].stats[0]
        accuracy_mean_ap_percentage = accuracy_mean_ap * 100.0
        summary_writer.add_scalar('val/accuracy (AP)% @ epoch', accuracy_mean_ap_percentage, global_step=epoch)
        # AP50
        accuracy_mean_ap50 = coco_evaluator.coco_eval['bbox'].stats[1]
        accuracy_mean_ap50_percentage = accuracy_mean_ap50 * 100.0
        summary_writer.add_scalar('val/accuracy (AP50)% @ epoch', accuracy_mean_ap50_percentage, global_step=epoch)
        # AP75
        accuracy_mean_ap75 = coco_evaluator.coco_eval['bbox'].stats[2]
        accuracy_mean_ap75_percentage = accuracy_mean_ap75 * 100.0
        summary_writer.add_scalar('val/accuracy (AP75)% @ epoch', accuracy_mean_ap75_percentage, global_step=epoch)

        # save image with boxes
        for saved_i, saved_r in zip(saved_images, saved_results):
            summary_tag = list(saved_r.keys())[0]
            summary_output = list(saved_r.values())[0]
            summary_image = saved_i.cpu()
            summary_boxes = summary_output['boxes'].cpu()
            summary_labels = summary_output['labels'].cpu()
            summary_scores = summary_output['scores'].cpu()
            summary_boxes = summary_boxes[summary_scores > 0.5].contiguous()
            summary_writer.add_image_with_boxes(f'val/image image_id:{summary_tag}', summary_image, summary_boxes, global_step=epoch)
        #
    #
    torch.set_num_threads(n_threads)
    return coco_evaluator


def export(args, model, model_name=None):
    if hasattr(args, 'quit_event') and args.quit_event.is_set():
        return
    #
    if not utils.is_main_process():
        return
    #

    # export onnx model
    model.eval()

    model = model.module if utils.is_parallel_module(model) else model
    model = model.module if utils.is_quant_module(model) else model
    assert hasattr(model, 'configure_forward'), 'writing partial onnx without post process is not supported by this model'

    # cpu based export is more reliable - so convert to cpu
    model_copy = copy.deepcopy(model).cpu()

    input_names = ['images']
    output_names = ['boxes', 'scores', 'labels']
    # TODO: why this is different from the onnx output
    output_names_proto = ['boxes', 'labels', 'scores']

    device = next(model_copy.parameters()).device
    model_name = 'model' if model_name is None else model_name
    image_size_tuple = args.input_size if isinstance(args.input_size, (list,tuple)) else \
        (args.input_size, args.input_size)

    data_shape = (1,3,*image_size_tuple)
    example_input = torch.rand(*data_shape).to(device)

    output_onnx_file = os.path.join(args.output_dir, 'model.onnx')

    # configure_forward is used to export a model without preprocess
    model_copy.configure_forward(with_preprocess=False)

    torch.onnx.export(
        model_copy,
        example_input,
        output_onnx_file,
        input_names=input_names,
        output_names=output_names,
        # export_params=True,
        # keep_initializers_as_inputs=True,
        # do_constant_folding=False,
        # verbose=False,
        # training=torch.onnx.TrainingMode.PRESERVE,
        opset_version=args.opset_version)
    # shape inference to make it easy for inference
    onnx.shape_inference.infer_shapes_path(output_onnx_file, output_onnx_file)

    # export prototxt file for detection model_copy without preprocess or postprocess
    model_copy.configure_forward(with_preprocess=False, with_postprocess=False)
    # this output_partial_onnx_file is only for temporary purposes
    output_onnx_file_ext = os.path.splitext(output_onnx_file)
    output_onnxproto_file = output_onnx_file_ext[0] + '-proto' + output_onnx_file_ext[1]
    export_proto.export_model_proto(dict(), model_copy, example_input, output_onnx_file, output_onnxproto_file,
                    output_names=output_names_proto, opset_version=args.opset_version)
    # shape inference to make it easy for inference
    onnx.shape_inference.infer_shapes_path(output_onnxproto_file, output_onnxproto_file)

    # restore the full model with pre and post process
    model_copy.configure_forward()

    # #export torchscript model - disabled for time being
    # script_model = torch.jit.trace(model_copy, example_input, strict=False)
    # torch.jit.save(script_model, os.path.join(args.output_dir, model_name+'_model.pth'))


def complexity(args, model):
    if hasattr(args, 'quit_event') and args.quit_event.is_set():
        return
    #
    model.eval()
    image_size_tuple = args.input_size if isinstance(args.input_size, (list,tuple)) else \
        (args.input_size, args.input_size)
    data_shape = (1,3,*image_size_tuple)
    # here next() is same as using list()[0], but more efficient,
    # since it doesn't convert the whole generator to list
    device = next(model.parameters()).device
    try:
        torchinfo.summary(model, data_shape, depth=10, device=device)
    except UnicodeEncodeError:
        warnings.warm('torchinfo.summary could not print - please check the language/locale')

