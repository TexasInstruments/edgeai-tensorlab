import math
from typing import List, Dict, Any

import torch.nn as nn
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.models.layers.yolo_layers import AConv, RepNCSPELAN, Conv, ELAN

class Aconv_RepNCSPELAN(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 part_channel,
                 csp_arg: Dict[str, Any] = {},):
        super().__init__()
        self.aconv = AConv(in_channel,out_channel)
        self.repncspelan = RepNCSPELAN(out_channel,out_channel,part_channels=part_channel,
                                       csp_args=csp_arg)
    
    def forward(self, x):
        x = self.aconv(x)
        return self.repncspelan(x)

@MODELS.register_module()
class YOLOV9Backbone(BaseModule):
    def __init__(self, 
                    init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')
                     ):
        super().__init__(init_cfg)
        self.conv1 = Conv(in_channels=3, out_channels=32, kernel_size=3,stride=2)
        self.conv2 = Conv(in_channels=32, out_channels=64, kernel_size=3,stride=2)
        self.elan = ELAN(in_channels=64, out_channels=64, part_channels=64)

        self.aconv_repncspelan1 = Aconv_RepNCSPELAN(in_channel=64,out_channel=128,part_channel=128,
                                                   csp_arg={"repeat_num": 3})
        self.aconv_repncspelan2 = Aconv_RepNCSPELAN(in_channel=128,out_channel=192,part_channel=192,
                                                   csp_arg={"repeat_num": 3})
        self.aconv_repncspelan3 = Aconv_RepNCSPELAN(in_channel=192,out_channel=256,part_channel=256,
                                                   csp_arg={"repeat_num": 3})
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.elan(x)
        b3 = self.aconv_repncspelan1(x)
        b4 = self.aconv_repncspelan2(b3)
        b5 = self.aconv_repncspelan3(b4)
        return (b3,b4,b5)