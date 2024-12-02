from typing import Sequence
import math

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.models.layers.yolo_layers import SPPELAN,UpSample,RepNCSPELAN, AConv


@MODELS.register_module()
class YOLOV9Neck(BaseModule):
    def __init__(self,
                 in_channels:Sequence[int] =[128, 192, 256],
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
        # self.aconv_layers = nn.ModuleList()
        # self.repncspelan_layers2 = nn.ModuleList()

        for idx in range(len(in_channels) - 1, 0, -1):
            self.repncspelan_layers.append(
                RepNCSPELAN(in_channels=in_channels[idx]+in_channels[idx-1],
                            out_channels=in_channels[idx-1],
                            part_channels=in_channels[idx-1],
                            csp_args=csp_arg
                            )
            )

        # for idx in range(len(in_channels)-1):
        #     self.aconv_layers.append(
        #         AConv(in_channels[idx], in_channels[idx+1]//2)
        #     )
        #     self.repncspelan_layers2.append(
        #         RepNCSPELAN(in_channels=in_channels[idx+1]+in_channels[idx+1]//2,
        #                     out_channels=in_channels[idx+1],
        #                     part_channels=in_channels[idx+1],
        #                     csp_args=csp_arg
        #                     )
        #     )

        #removed for-loop to load weights correctly
        #0
        idx = 0
        self.aconv_layer_0 = AConv(in_channels[idx], in_channels[idx+1]//2)
        self.repncspelan_layers2_0 = RepNCSPELAN(
                            in_channels=in_channels[idx+1]+in_channels[idx+1]//2,
                            out_channels=in_channels[idx+1],
                            part_channels=in_channels[idx+1],
                            csp_args=csp_arg
                            )
        #1
        idx=1
        self.aconv_layer_1 = AConv(in_channels[idx], in_channels[idx+1]//2)
        self.repncspelan_layers2_1 = RepNCSPELAN(
                            in_channels=in_channels[idx+1]+in_channels[idx+1]//2,
                            out_channels=in_channels[idx+1],
                            part_channels=in_channels[idx+1],
                            csp_args=csp_arg
                            )
            
        # for idx in range(len(in_channels)-1):
        #     self.aconv_layers.append(
        #         AConv(in_channels[idx], in_channels[idx+1]//2)
        #     )
        #     self.repncspelan_layers2.append(
        #         RepNCSPELAN(in_channels=in_channels[idx+1]+in_channels[idx+1]//2,
        #                     out_channels=in_channels[idx+1],
        #                     part_channels=in_channels[idx+1],
        #                     csp_args=csp_arg
        #                     )
        #     )



    def forward(self, inputs):

        outs = []
        inner_outs = []
        inner_outs.append(self.sppelan(inputs[-1]))
        
        for idx in range(len(self.in_channels) - 1, 0, -1):
            x = torch.cat([self.upsample(inner_outs[-1]),inputs[idx-1]],1)
            inner_outs.append(self.repncspelan_layers[len(self.in_channels) - 1 - idx](x))
        outs.append(inner_outs[-1])
        
        # for idx in range(len(self.in_channels)-1):
        #     x = torch.cat([self.aconv_layers[idx](outs[-1]),inner_outs[1-idx]],1)
        #     outs.append(self.repncspelan_layers2[idx](x))

        #removed for-loop to load weights correctly
        idx=0
        x = torch.cat([self.aconv_layer_0(outs[-1]),inner_outs[1-idx]],1)
        outs.append(self.repncspelan_layers2_0(x))

        idx=1
        x = torch.cat([self.aconv_layer_1(outs[-1]),inner_outs[1-idx]],1)
        outs.append(self.repncspelan_layers2_1(x))
        
        return outs


