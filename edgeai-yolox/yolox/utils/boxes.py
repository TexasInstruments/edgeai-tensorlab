#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np

import torch
import torch.nn as nn
import torchvision

__all__ = [
    "filter_box",
    "postprocess",
    "postprocess_object_pose",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "adjust_kpts_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "cxcywh2xyxy",
    "PostprocessExport",
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False, human_pose=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        if not human_pose:
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        else:
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred, kpts)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 6:]), 1)

        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

def postprocess_object_pose(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, R, T, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], image_pred[:, -9:], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, -2],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, -2],
                detections[:, -1],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def postprocess_export(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, task=None):
    """
    This function is called while exporting an OD model or human-pose estimation model. Output of the ONNX model is ensured to match the TIDL output in float mode.
    """
    cx, cy, w, h = prediction[..., 0:1], prediction[..., 1:2], prediction[..., 2:3], prediction[..., 3:4]
    box = cxcywh2xyxy_export(cx, cy, w, h)
    prediction[:, :, :4] = box

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf =  image_pred[:, 4:5] * class_conf
        conf_mask = (conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        tensor_cat_inp = [image_pred[:, :4], conf, class_pred.float()]
        if task == "human_pose":
            tensor_cat_inp.extend([image_pred[:,6:]])
        detections = torch.cat(tensor_cat_inp, 1)

        detections = detections[conf_mask]
        # manu: commenting out to avoid onnx export error.
        # if not detections.size(0):
        #     continue
        class_2d_offset = detections[:, -1:] * 4096  # class_2d_offser

        nms_out_index = torchvision.ops.nms(
            detections[:, :4] + class_2d_offset,
            detections[:, 4],
            nms_thre,
        )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

def postprocess_export_object_pose(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, camera_matrix=None):
    """
    This function is called while exporting an ObjectPose model. Output of the ONNX model is ensured to match the TIDL output in float mode.
    """
    cx, cy, w, h = prediction[..., 0:1], prediction[..., 1:2], prediction[..., 2:3], prediction[..., 3:4]
    box = cxcywh2xyxy_export(cx, cy, w, h)
    prediction[:, :, :4] = box

    # tx = ((pose[11] / r_w) - camera_matrix[2]) * tz / camera_matrix[0]
    # ty = ((pose[12] / r_h) - camera_matrix[5]) * tz / camera_matrix[4]


    # Transform the translation vector in expected format
    tx, ty, tz = prediction[..., -3], prediction[..., -2], prediction[..., -1]
    tz *= 100
    tx = (tx - camera_matrix[2]) * tz / camera_matrix[0]
    ty = (ty - camera_matrix[5]) * tz / camera_matrix[4]
    prediction[..., -3], prediction[..., -2], prediction[..., -1] = tx, ty, tz

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf = image_pred[:, 4:5] * class_conf
        conf_mask = (conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, R, T, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :4], conf, class_pred.float(), image_pred[:, -9:]), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue
        class_2d_offset = detections[:, 5:6] * 4096  # class_2d_offser

        nms_out_index = torchvision.ops.nms(
            detections[:, :4] + class_2d_offset,
            detections[:, 4] ,
            nms_thre,
        )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output



def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def adjust_kpts_anns(kpts, scale_ratio, padw, padh, w_max, h_max):
    kpts[:, 0::2][kpts[:, 0::2]!=0] = kpts[:, 0::2][kpts[:, 0::2]!=0] * scale_ratio + padw
    kpts[:, 1::2][kpts[:, 1::2]!=0] = kpts[:, 1::2][kpts[:, 1::2]!=0] * scale_ratio + padh
    kpts[:, 0::2] = np.clip(kpts[:, 0::2] , 0, w_max)
    kpts[:, 1::2] = np.clip(kpts[:, 1::2] , 0, h_max)
    return kpts


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def cxcywh2xyxy(bboxes):
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # top left x
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # top left y
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # bottom right x
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # bottom right y
    return bboxes

def cxcywh2xyxy_export(cx,cy,w,h):
    #This function is used while exporting ONNX models
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    halfw = w/2
    halfh = h/2
    xmin = cx - halfw  # top left x
    ymin = cy - halfh  # top left y
    xmax = cx + halfw  # bottom right x
    ymax = cy + halfh  # bottom right y
    return torch.cat((xmin, ymin, xmax, ymax), 2)


class PostprocessExport(nn.Module):
    def __init__(self, conf_thre=0.7, nms_thre=0.45, num_classes=80, object_pose=False, camera_matrix=None, task=None):
        super(PostprocessExport, self).__init__()
        self.conf_thre = conf_thre
        self.nms_thre = nms_thre
        self.num_classes = num_classes
        self.object_pose = object_pose
        self.camera_matrix = camera_matrix
        self.task = task

    def forward(self, prediction):
        if not self.object_pose:
            return postprocess_export(prediction, self.num_classes, conf_thre=self.conf_thre, nms_thre=self.nms_thre, task=self.task)
        else:
            return postprocess_export_object_pose(prediction, self.num_classes, conf_thre=self.conf_thre, nms_thre=self.nms_thre, camera_matrix=self.camera_matrix)

