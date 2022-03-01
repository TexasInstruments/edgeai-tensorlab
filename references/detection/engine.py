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
import copy

import torchvision.models.detection.mask_rcnn

from torchvision.edgeailite import xnn
import cv2
import numpy as np

basename = os.path.splitext(os.path.basename(__file__))[0]
if __name__.startswith(basename):
    from coco_utils import get_coco_api_from_dataset
    from coco_eval import CocoEvaluator
    import utils
    import export_proto
else:
    from .coco_utils import get_coco_api_from_dataset
    from .coco_eval import CocoEvaluator
    from . import utils
    from . import export_proto
#

def train_one_epoch(args, model, optimizer, data_loader, device, epoch, print_freq, summary_writer=None, anno=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = min(print_freq, len(data_loader))

    # warmup was earlier here, but is now included in the original lr_scheduler
    # see creating of lr_scheduler in train.py
    for batch_id, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if batch_id > args.max_batches:
            break
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        draw_boxes_training(args=args, images=images, targets=targets, anno=anno)

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

        if hasattr(args, 'quit_event') and args.quit_event is not None and args.quit_event.is_set():
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

def draw_boxes(args=None, images=None, outputs=None, img_name=None):
    if args.save_imgs_path is not None and args.test_only:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        fontColor = (255, 255, 255)
        lineType = 2

        grid = torchvision.utils.make_grid(images[0])
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        image_nd_array = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        cv2_image = image_nd_array[:, :, ::-1].copy()

        # draw boxes
        for box, score, label in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels']):
            if score > 0.2:
                x1, y1, x2, y2 = box
                pts = np.array([x1, y1, x2, y1, x2, y2, x1, y2], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(cv2_image, [pts], True, (255, 0, 0), thickness=2)

                # display text
                txt_to_disp = "{:02d}_{:4.2f}".format(label.cpu().numpy(), score.cpu().numpy())
                offset = 10
                y = pts[0, 0, 1] - offset if pts[0, 0, 1] > offset else pts[0, 0, 1] + offset
                coord = (pts[0, 0, 0], y)
                cv2.putText(cv2_image, txt_to_disp, coord, font, fontScale, fontColor, lineType)
        img_name = os.path.join(args.save_imgs_path, img_name)
        os.makedirs(os.path.dirname(img_name), exist_ok=True)
        cv2.imwrite(img_name, cv2_image)
    return

def draw_boxes_training(args=None, targets=None, images=None, anno=None, en_text_to_display=False):
    if args.save_imgs_path is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        fontColor = (255, 255, 255)
        lineType = 2

        for (image, target) in zip(images, targets):
            grid = torchvision.utils.make_grid(image)
            # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
            image_nd_array = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            cv2_image = image_nd_array[:, :, ::-1].copy()

            # draw boxes
            scores = [1.0]*len(target['boxes'])
            for box, score, label in zip(target['boxes'], scores, target['labels']):
                if score > 0.2:
                    x1, y1, x2, y2 = box
                    pts = np.array([x1, y1, x2, y1, x2, y2, x1, y2], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(cv2_image, [pts], True, (255, 0, 0), thickness=2)

                    # display text
                    if en_text_to_display:
                        txt_to_disp = "{:02d}_{:4.2f}".format(label.cpu().numpy(), score)
                        offset = 10
                        y = pts[0, 0, 1] - offset if pts[0, 0, 1] > offset else pts[0, 0, 1] + offset
                        coord = (pts[0, 0, 0], y)
                        cv2.putText(cv2_image, txt_to_disp, coord, font, fontScale, fontColor, lineType)

            img_name = id_to_filename(anno=anno, image_id=target['image_id'], args=None)
            img_name = os.path.join(args.save_imgs_path, img_name)
            os.makedirs(os.path.dirname(img_name), exist_ok=True)
            cv2.imwrite(img_name, cv2_image)
    return

def id_to_filename(anno=None, image_id=None, args=None):
    found = None
    for img in anno['images']:
        if img['id'] == image_id:
            found = img['file_name']
            break
    return found

def write_outputs_txt_format(args=None, outputs=None, img_name=None):
    if args.save_op_txt_path is not None and args.test_only:
        os.makedirs(args.save_op_txt_path, exist_ok=True)
        ext = os.path.splitext(img_name)[1]
        filename = img_name.replace(ext, '.txt')
        filename = os.path.join(args.save_op_txt_path, filename)
        scores = np.expand_dims(outputs[0]['scores'].cpu().numpy(), axis=-1)
        result = np.hstack((outputs[0]['boxes'].cpu().numpy(), scores))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        header = "{} \n{:04d}".format(os.path.basename(filename), scores.shape[0])
        np.savetxt(filename, result, header=header)
    return

# Visualize results as well as for writing detected boxes in text format
def store_boxes(args=None, targets=None, outputs=None, images=None, anno=None):
    if args.save_imgs_path or args.save_op_txt_path:
        image_id = targets[0]['image_id'][0].cpu().numpy()
        img_name = id_to_filename(anno=anno, image_id=image_id, args=args)

    if args.save_imgs_path is not None and args.test_only:
        draw_boxes(args=args, images=images, outputs=outputs, img_name=img_name)
    
    if args.save_op_txt_path is not None and args.test_only:
        write_outputs_txt_format(args=args, outputs=outputs, img_name=img_name)
    return

def convert_results_anno_for_wider_face_eval(outputs=None, targets=None, min_size=0):
    result_wider_face_format = torch.hstack((outputs[0]['boxes'].cpu(), torch.unsqueeze(outputs[0]['scores'], 1).cpu())).numpy()
    annotations = dict()

    annotations['bboxes'] = []
    annotations['bboxes_ignore'] = []
    for box in targets[0]['boxes']:
        if (box[3]-box[1]) >= min_size or (box[2]-box[0]) >= min_size:
            annotations['bboxes'].append(box.cpu().numpy())
        else:
            annotations['bboxes_ignore'].append(box.cpu().numpy())

    annotations['bboxes'] = np.array(annotations['bboxes'])
    annotations['labels'] = np.array([0] * len(annotations['bboxes']))
    n_boxes = len(annotations['bboxes'])
    if n_boxes == 0:
        annotations['bboxes'] = np.zeros((0, 4))

    annotations['bboxes_ignore'] = np.array(annotations['bboxes_ignore'])
    n_ignore_boxes = len(annotations['bboxes_ignore'])
    annotations['labels_ignore'] = np.array([0] * n_ignore_boxes)
    if n_ignore_boxes == 0:
        annotations['bboxes_ignore'] = None
        annotations['labels_ignore'] = None

    return result_wider_face_format, annotations

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
    saved_images = collections.deque(maxlen=5)
    saved_results = collections.deque(maxlen=5)
    batch_counter = 0
    print_freq = min(print_freq, len(data_loader))
    results_wider_face_format = []
    annotations_wider_face_format = []
    anno = None
    if args.save_imgs_path or args.save_op_txt_path:
        import json
        anno = json.load(open(os.path.join(args.data_path, 'annotations', 'instances_val.json')))
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        if batch_counter > args.max_batches:
            break
        images = list(img.to(device) for img in images)

        if synchronize_time and torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        if args.en_wider_face_eval:
            result_wider_face_format, annotation_wider_face_format = convert_results_anno_for_wider_face_eval(outputs=outputs, targets=targets)
            results_wider_face_format.append([result_wider_face_format])
            annotations_wider_face_format.append(annotation_wider_face_format)
        store_boxes(args=args, targets=targets, outputs=outputs, images=images, anno=anno)

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

        if hasattr(args, 'quit_event') and args.quit_event is not None and args.quit_event.is_set():
            break

        batch_counter += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()
    if args.en_wider_face_eval:
        import vedadet.misc.evaluation.mean_ap as wider_face_mean_ap
        mean_ap, _ = wider_face_mean_ap.eval_map(
                results_wider_face_format,
                annotations_wider_face_format,
                scale_ranges=None,
                iou_thr=0.5,
                dataset=('face',),
                logger=None)

    if hasattr(args, 'quit_event') and args.quit_event is not None and args.quit_event.is_set():
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
            summary_tag = random.randrange(5)
            summary_output = list(saved_r.values())[0]
            summary_image = saved_i.cpu()
            summary_boxes = summary_output['boxes'].cpu()
            summary_labels = summary_output['labels'].cpu()
            summary_scores = summary_output['scores'].cpu()
            summary_boxes = summary_boxes[summary_scores > 0.2].contiguous()
            summary_writer.add_image_with_boxes(f'val/image image_id:{summary_tag}', summary_image, summary_boxes, global_step=epoch)
        #
    #
    torch.set_num_threads(n_threads)
    return coco_evaluator


def export(args, model, model_name=None):
    if hasattr(args, 'quit_event') and args.quit_event is not None and args.quit_event.is_set():
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
    export_proto.export_model_proto(args, model_copy, example_input, output_onnx_file,
                                    input_names, output_names, output_names_proto,
                                    args.opset_version)

    # #export torchscript model - disabled for time being
    # script_model = torch.jit.trace(model_copy, example_input, strict=False)
    # torch.jit.save(script_model, os.path.join(args.output_dir, model_name+'_model.pth'))


def complexity(args, model):
    if hasattr(args, 'quit_event') and args.quit_event is not None and args.quit_event.is_set():
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
    except Exception as err:
        warnings.warn(f'torchinfo.summary could not print - {str(err)}')
