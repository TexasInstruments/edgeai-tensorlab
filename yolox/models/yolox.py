#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from yolox.models.yolo_object_pose_head import YOLOXObjectPoseHead
import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_kpts_head import YOLOXHeadKPTS
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            if isinstance(self.head, YOLOXHead):
                loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                    fpn_outs, targets, x
                )
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "num_fg": num_fg,
                }
            elif isinstance(self.head, YOLOXObjectPoseHead):
                loss, iou_loss, conf_loss, cls_loss, rot_loss, trn_xy_loss, trn_xyz_loss, l1_loss, adds_loss, num_fg = self.head(
                    fpn_outs, targets, x
                )
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "rot_loss": rot_loss,
                    "trn_xy_loss": trn_xy_loss,
                    "trn_xyz_loss": trn_xyz_loss,
                    "adds_loss": adds_loss,
                    "num_fg": num_fg,
                }
            elif isinstance(self.head, YOLOXHeadKPTS):
                loss, iou_loss, conf_loss, cls_loss, l1_loss, kpts_loss, kpts_vis_loss, loss_l1_kpts, num_fg = self.head(
                    fpn_outs, targets, x
                )
                outputs = {
                    "total_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    "kpts_loss": kpts_loss,
                    "kpts_vis_loss": kpts_vis_loss,
                    "l1_loss_kpts": loss_l1_kpts,
                    "num_fg": num_fg,
                }

        else:
            outputs = self.head(fpn_outs)

        return outputs
