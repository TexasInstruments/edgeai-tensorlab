# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, normal_init
from mmcv.runner import force_fp32
from mmcv.cnn import constant_init, kaiming_init

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, images_to_levels,
                        multi_apply, multiclass_nms)
from mmdet.models.builder import HEADS
from mmdet.models.losses import smooth_l1_loss
from mmdet.models.dense_heads.yolo_head import YOLOV3Head
from ...ops import ConvModuleWrapper


@HEADS.register_module()
class YOLOV3LiteHead(YOLOV3Head):
    """YOLOV3LiteHead Paper link: https://arxiv.org/abs/1804.02767.
    """
    def __init__(self, *args, **kwargs):
        super(YOLOV3LiteHead, self).__init__(*args, **kwargs)

    def _init_layers(self):
        self.convs_bridge = nn.ModuleList()
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_bridge = ConvModuleWrapper(
                self.in_channels[i],
                self.out_channels[i],
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            conv_pred = nn.Conv2d(self.out_channels[i],
                                  self.num_anchors * self.num_attrib, 1)

            self.convs_bridge.append(conv_bridge)
            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1)
