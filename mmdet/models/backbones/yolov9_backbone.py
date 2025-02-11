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


import math
from typing import List, Dict, Any, Sequence

import torch.nn as nn
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.models.layers.yolo_layers import AConv, RepNCSPELAN, Conv, ELAN

class Aconv_RepNCSPELAN(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 part_channel,
                 pool_kernel_size=2,
                 csp_arg: Dict[str, Any] = {},):
        super().__init__()
        self.aconv = AConv(in_channel,out_channel,pool_kernel_size)
        self.repncspelan = RepNCSPELAN(out_channel,out_channel,part_channels=part_channel,
                                       csp_args=csp_arg)
    
    def forward(self, x):
        x = self.aconv(x)
        return self.repncspelan(x)

@MODELS.register_module()
class YOLOV9Backbone(BaseModule):
    def __init__(self,
                    stem_channels : Sequence[int] = [16, 32],
                    expand_list:Sequence[int] = [64, 96, 128],
                    pool_kernel_size:int = 2,
                    init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')
                     ):
        super().__init__(init_cfg)
        self.conv1 = Conv(in_channels=3, out_channels=stem_channels[0], kernel_size=3,stride=2)
        self.conv2 = Conv(in_channels=stem_channels[0], out_channels=stem_channels[1], kernel_size=3,stride=2)
        self.elan = ELAN(in_channels=stem_channels[1], out_channels=stem_channels[1], part_channels=stem_channels[1])

        self.aconv_repncspelan1 = Aconv_RepNCSPELAN(in_channel=stem_channels[1],
                                                    out_channel=expand_list[0],
                                                    part_channel=expand_list[0],
                                                    pool_kernel_size=pool_kernel_size,
                                                    csp_arg={"repeat_num": 3})
        self.aconv_repncspelan2 = Aconv_RepNCSPELAN(in_channel=expand_list[0],
                                                    out_channel=expand_list[1],
                                                    part_channel=expand_list[1],
                                                    pool_kernel_size=pool_kernel_size,
                                                    csp_arg={"repeat_num": 3})
        self.aconv_repncspelan3 = Aconv_RepNCSPELAN(in_channel=expand_list[1],
                                                    out_channel=expand_list[2],
                                                    part_channel=expand_list[2],
                                                    pool_kernel_size=pool_kernel_size,
                                                    csp_arg={"repeat_num": 3})
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.elan(x)
        b3 = self.aconv_repncspelan1(x)
        b4 = self.aconv_repncspelan2(b3)
        b5 = self.aconv_repncspelan3(b4)
        return (b3,b4,b5)