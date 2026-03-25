# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from torch import nn
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet3d.structures.bbox_3d.utils import limit_period

class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

