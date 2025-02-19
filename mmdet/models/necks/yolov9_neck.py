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


from typing import Sequence
import math

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.models.layers.yolo_layers import SPPELAN,UpSample,RepNCSPELAN,AConv


@MODELS.register_module()
class YOLOV9Neck(BaseModule):
    def __init__(self,
                 in_channels:Sequence[int] = [64, 96, 128],
                 pool_kernel_size:int = 2,
                 pool_type = 'max',
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 csp_arg = {"repeat_num": 3},
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')
                     ):
        super().__init__(init_cfg)
        self.in_channels = in_channels

        self.sppelan = SPPELAN(in_channels[-1],out_channels=in_channels[-1])
        self.upsample = UpSample(**upsample_cfg)
        self.repncspelan_layers = nn.ModuleList()

        for idx in range(len(in_channels) - 1, 0, -1):
            self.repncspelan_layers.append(
                RepNCSPELAN(in_channels=in_channels[idx]+in_channels[idx-1],
                            out_channels=in_channels[idx-1],
                            part_channels=in_channels[idx-1],
                            csp_args=csp_arg
                            )
            )
        self.aconv_layer_0 = AConv(in_channels[0], in_channels[1]//2, pool_kernel_size=pool_kernel_size,pool_type=pool_type)
        self.repncspelan_layers2_0 = RepNCSPELAN(
                            in_channels=in_channels[1]+in_channels[1]//2,
                            out_channels=in_channels[1],
                            part_channels=in_channels[1],
                            csp_args=csp_arg
                            )
        self.aconv_layer_1 = AConv(in_channels[1], in_channels[2]//2, pool_kernel_size=pool_kernel_size, pool_type=pool_type)
        self.repncspelan_layers2_1 = RepNCSPELAN(
                            in_channels=in_channels[2]+in_channels[2]//2,
                            out_channels=in_channels[2],
                            part_channels=in_channels[2],
                            csp_args=csp_arg
                            )

    def forward(self, inputs):

        outs = []
        inner_outs = []
        inner_outs.append(self.sppelan(inputs[-1]))
        
        for idx in range(len(self.in_channels) - 1, 0, -1):
            x = torch.cat([self.upsample(inner_outs[-1]),inputs[idx-1]],1)
            inner_outs.append(self.repncspelan_layers[len(self.in_channels) - 1 - idx](x))
        outs.append(inner_outs[-1])
        
        x = torch.cat([self.aconv_layer_0(outs[-1]),inner_outs[1]],1)
        outs.append(self.repncspelan_layers2_0(x))

        x = torch.cat([self.aconv_layer_1(outs[-1]),inner_outs[0]],1)
        outs.append(self.repncspelan_layers2_1(x))
        
        return outs


