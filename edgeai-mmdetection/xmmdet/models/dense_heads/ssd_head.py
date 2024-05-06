import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, multi_apply)
from mmdet.models.builder import HEADS
from mmdet.models.losses import smooth_l1_loss
from mmdet.models.dense_heads.ssd_head import SSDHead
from ...ops import ConvModuleWrapper


# TODO: add loss evaluator for SSD
@HEADS.register_module()
class SSDLiteHead(SSDHead):

    def __init__(self, *args, **kwargs):
        conv_cfg = kwargs.pop('conv_cfg', None) # not supported in base class - so pop it
        super(SSDLiteHead, self).__init__(*args, **kwargs)
        num_anchors = self.anchor_generator.num_base_anchors

        reg_convs = []
        cls_convs = []
        for i in range(len(self.in_channels)):
            reg_convs.append(
                ConvModuleWrapper(
                    self.in_channels[i],
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=None,
                    act_cfg=None))
            cls_convs.append(
                ConvModuleWrapper(
                    self.in_channels[i],
                    num_anchors[i] * (self.num_classes + 1),
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=None,
                    act_cfg=None))
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)
