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
from mmdet.models.layers.yolo_layers import (SPPCSPConv, SPPCSPTinyConv, Conv,
                                             RepConv, Pool, UpSample)


class Conv_Upsample(nn.Module):
    def __init__(self,
                 in_channel:int
                 ):
        super().__init__()
        self.conv1 = Conv(in_channels=in_channel,
                          out_channels=in_channel//2,
                          kernel_size=1)
        self.upsample = UpSample(scale_factor=2)
        self.conv2 = Conv(in_channels=in_channel*2,
                          out_channels=in_channel//2,
                          kernel_size=1)
        
    def forward(self, x):
        y = self.upsample(self.conv1(x[0]))
        return torch.cat([self.conv2(x[1]), y], dim=1)
    
class Conv_Upsample_Tiny(nn.Module):
    def __init__(self,
                 in_channel:int
                 ):
        super().__init__()
        self.conv1 = Conv(in_channels=in_channel,
                          out_channels=in_channel//2,
                          kernel_size=1)
        self.upsample = UpSample(scale_factor=2)
        self.conv2 = Conv(in_channels=in_channel,
                          out_channels=in_channel//2,
                          kernel_size=1)
        
    def forward(self, x):
        y =  self.upsample(self.conv1(x[0]))
        return torch.cat([self.conv2(x[1]), y], dim=1)
    

class Elan_Neck(nn.Module):
    def __init__(self, in_channel: int):
        super().__init__()
        out_channel = in_channel // 2
        intermediate_channel = out_channel // 2
        concat_channel = in_channel * 2
        self.conv1 = Conv(in_channels=in_channel,
                          out_channels=out_channel,
                          kernel_size=1)
        self.conv2 = Conv(in_channels=in_channel,
                          out_channels=out_channel,
                          kernel_size=1)
        self.conv3 = Conv(in_channels=out_channel,
                          out_channels=intermediate_channel,
                          kernel_size=3)
        self.conv4 = Conv(in_channels=intermediate_channel,
                          out_channels=intermediate_channel,
                          kernel_size=3)
        self.conv5 = Conv(in_channels=intermediate_channel,
                          out_channels=intermediate_channel,
                          kernel_size=3)
        self.conv6 = Conv(in_channels=intermediate_channel,
                          out_channels=intermediate_channel,
                          kernel_size=3)
        self.conv_out = Conv(in_channels=concat_channel,
                          out_channels=out_channel,
                          kernel_size=1)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        return self.conv_out(torch.cat([x6, x5, x4, x3, x2, x1], dim=1))
    
class TinyDownSampleNeck(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        mid_ch = in_ch // 4
        self.conv1 = Conv(in_channels=in_ch,
                        out_channels=mid_ch,
                        kernel_size=1,
                        stride=1)
        self.conv2 = Conv(in_channels=in_ch,
                        out_channels=mid_ch,
                        kernel_size=1,
                        stride=1)
        self.conv3 = Conv(in_channels=mid_ch,
                        out_channels=mid_ch,
                        kernel_size=3,
                        stride=1)
        self.conv4 = Conv(in_channels=mid_ch,
                        out_channels=mid_ch,
                        kernel_size=3,
                        stride=1)
        self.conv_out = Conv(in_channels=in_ch,
                        out_channels=mid_ch*2,
                        kernel_size=1,
                        stride=1)
        
    def forward(self, x):
        x1 = self.conv1(x)
        y1 = self.conv2(x)
        y2 = self.conv3(y1)
        y3 = self.conv4(y2)
        return self.conv_out(torch.cat([y3, y2, y1, x1],dim=1))
        

class Pool_Conv_Neck(nn.Module):
    def __init__(self,
                 out_channel:int):
        super().__init__()
        self.pool = Pool(padding=0)
        self.conv1 = Conv(in_channels=out_channel, 
                          out_channels=out_channel,
                          kernel_size=1)
        self.conv2 = Conv(in_channels=out_channel,
                          out_channels=out_channel,
                          kernel_size=1)
        self.conv3 = Conv(in_channels=out_channel,
                          out_channels=out_channel,
                          kernel_size=3,
                          stride=2)
        
    def forward(self, x):
        return torch.cat([self.conv3(self.conv2(x[0])), self.conv1(self.pool(x[0])), x[1]], dim=1)
    
class Conv_Downsample_Tiny(nn.Module):
    def __init__(self,
                 out_channel:int):
        super().__init__()
        self.conv1 = Conv(in_channels=out_channel//2,
                          out_channels=out_channel,
                          kernel_size=3,
                          stride=2)
        
    def forward(self, x):
        return torch.cat([self.conv1(x[0]), x[1]], dim=1)
    

@MODELS.register_module()
class YOLOV7Neck(BaseModule):
    def __init__(self,
                 top_down_channels: Sequence[int] = [512, 256],
                 down_sample_channels: Sequence[int] = [256, 512],
                 output_channels:Sequence[int] =[256, 512, 1024],
                #  upsample_cfg=dict(scale_factor=2, mode='nearest'),
                #  csp_arg = {"repeat_num": 3},
                init_cfg=None
                     ):
        super().__init__(init_cfg)
        self.top_down_channels=top_down_channels
        self.down_sample_channels=down_sample_channels
        self.output_channels=output_channels

        self.sppcspconv = SPPCSPConv(in_channels=1024, out_channels=512)

        #top_down_layers
        self.top_down_layers = nn.ModuleList()
        for idx in range(len(self.down_sample_channels), 0, -1):
            self.top_down_layers.append(
                nn.Sequential(
                    Conv_Upsample(down_sample_channels[idx-1]),
                    Elan_Neck(in_channel=down_sample_channels[idx-1])
                )
            )

        #down_sample_layers
        self.down_sample_layers = nn.ModuleList()
        for i in range(len(down_sample_channels)):
            self.down_sample_layers.append(
                nn.Sequential(
                    Pool_Conv_Neck(down_sample_channels[i]//2),
                    Elan_Neck(down_sample_channels[i]*2)
                )
            )

        #out_layers
        self.out_layers = nn.ModuleList()
        for i in range(len(output_channels)):
            self.out_layers.append(RepConv(in_channels=output_channels[i]//2,
                                           out_channels=output_channels[i]))
            
    def forward(self, inputs):
        top_down_outs = []
        down_sample_outs = []
        outs = []

        top_down_outs.append(self.sppcspconv(inputs[-1]))
        for idx in range(len(self.down_sample_channels), 0, -1):
            top_down_outs.append(
                self.top_down_layers[len(self.down_sample_channels)-idx](
                    [top_down_outs[-1],
                     inputs[idx-1]]
                )
            )
        down_sample_outs.append(top_down_outs[-1])
        for idx in range(len(self.down_sample_channels)):
            down_sample_outs.append(
                self.down_sample_layers[idx](
                    [down_sample_outs[-1],
                    top_down_outs[len(top_down_outs)-2-idx]])
            )
        for idx in range(len(self.output_channels)):
            outs.append(self.out_layers[idx](down_sample_outs[idx]))

        return outs
    

@MODELS.register_module()
class YOLOV7TinyNeck(BaseModule):
    def __init__(self,
                 top_down_channels: Sequence[int] = [256, 128],
                 down_sample_channels: Sequence[int] = [256, 512],
                 output_channels:Sequence[int] =[128, 256, 512],
                #  upsample_cfg=dict(scale_factor=2, mode='nearest'),
                #  csp_arg = {"repeat_num": 3},
                init_cfg=None
                     ):
        super().__init__(init_cfg)
        self.top_down_channels=top_down_channels
        self.down_sample_channels=down_sample_channels
        self.output_channels=output_channels

        self.sppcspconv = SPPCSPTinyConv(in_channels=512, out_channels=256)

        #top_down_layers
        self.top_down_layers = nn.ModuleList()
        for idx in range(len(self.top_down_channels)):
            self.top_down_layers.append(
                nn.Sequential(
                    Conv_Upsample_Tiny(top_down_channels[idx]),
                    TinyDownSampleNeck(in_ch=top_down_channels[idx])
                )
            )

        #down_sample_layers
        self.down_sample_layers = nn.ModuleList()
        for i in range(len(down_sample_channels)):
            self.down_sample_layers.append(
                nn.Sequential(
                    Conv_Downsample_Tiny(out_channel=down_sample_channels[i]//2),
                    TinyDownSampleNeck(in_ch=down_sample_channels[i])
                )
            )
        
        #out_layers
        self.out_layers = nn.ModuleList()
        for i in range(len(output_channels)):
            self.out_layers.append(
                Conv(in_channels=output_channels[i]//2,
                          out_channels=output_channels[i],
                          kernel_size=3,
                          stride=1),
            )
            
    def forward(self, inputs):
        top_down_outs = []
        down_sample_outs = []
        outs = []

        top_down_outs.append(self.sppcspconv(inputs[-1]))
        for idx in range(len(self.top_down_channels)):
            top_down_outs.append(
                self.top_down_layers[idx](
                    [top_down_outs[-1],
                     inputs[1-idx]]
                )
            )
        down_sample_outs.append(top_down_outs[-1])
        for idx in range(len(self.down_sample_channels)):
            down_sample_outs.append(
                self.down_sample_layers[idx](
                    [down_sample_outs[-1],
                    top_down_outs[len(top_down_outs)-2-idx]])
            )
        for idx in range(len(self.output_channels)):
            outs.append(self.out_layers[idx](down_sample_outs[idx]))

        return outs
