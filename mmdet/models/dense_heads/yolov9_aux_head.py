# Copyright (c) 2018-2025, Texas Instruments
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


from typing import List, Tuple, Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.layers.yolo_layers import SPPELAN,UpSample,RepNCSPELAN, AConv
from mmdet.models.layers.yolo_layers import Detection 


@MODELS.register_module()
class YOLOV9AuxHead(BaseModule):
    def __init__(self,
                 in_channels: Sequence[int] = [64, 96, 128],
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 csp_arg = {"repeat_num": 3},
                 num_classes: int = 80,
                 reg_max: int = 16,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')
                     ):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels

        #Neck_aux
        self.sppelan_aux = SPPELAN(in_channels[-1],out_channels=in_channels[-1])
        self.upsample_aux = UpSample(**upsample_cfg)
        
        self.repncspelan_layers_aux = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.repncspelan_layers_aux.append(
                RepNCSPELAN(in_channels=in_channels[idx]+in_channels[idx-1],
                            out_channels=in_channels[idx-1],
                            part_channels=in_channels[idx-1],
                            csp_args=csp_arg
                            )
            )

        #Head_aux
        self.heads_aux = nn.ModuleList(
            [Detection((in_channels[0], in_channel), num_classes, reg_max=reg_max) for in_channel in in_channels]
        )

    def forward(self, inputs):
        #auxiliary
        aux_outs = [None] * 3
        aux_outs[-1] = self.sppelan_aux(inputs[-1])
        for idx in range(len(self.in_channels) - 1, 0, -1):
            x = torch.cat([self.upsample_aux(aux_outs[idx]),inputs[idx-1]],1)
            aux_outs[idx-1] = self.repncspelan_layers_aux[len(self.in_channels) - 1 - idx](x)

        head_outs = []
        head_outs.append([head_aux(x) for x, head_aux in zip(aux_outs, self.heads_aux)])
        
        return head_outs