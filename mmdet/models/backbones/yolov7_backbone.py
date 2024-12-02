import math
from typing import List, Dict, Sequence
import torch.nn as nn
import torch

from mmdet.models.layers.yolo_layers import Conv, Concat, Pool
from mmengine.model import BaseModule
from mmdet.registry import MODELS

class Elan_BB(nn.Module):
    def __init__(self,
                    in_channel: int,
                    mid_channel:int,
                    out_channel:int
                     ):
        super().__init__()
        self.conv1 = Conv(in_channels=in_channel,
                          out_channels=mid_channel,
                          kernel_size=1)
        self.conv2 = Conv(in_channels=in_channel,
                          out_channels=mid_channel,
                          kernel_size=1)
        self.conv3 = Conv(in_channels=mid_channel,
                          out_channels=mid_channel,
                          kernel_size=3)
        self.conv4 = Conv(in_channels=mid_channel,
                          out_channels=mid_channel,
                          kernel_size=3)
        self.conv5 = Conv(in_channels=mid_channel,
                          out_channels=mid_channel,
                          kernel_size=3)
        self.conv6 = Conv(in_channels=mid_channel,
                          out_channels=mid_channel,
                          kernel_size=3)
        self.conv_out = Conv(in_channels=out_channel,
                             out_channels=out_channel,
                             kernel_size=1)
        
    def forward(self, x):
        x1 = self.conv1(x)
        y1 = self.conv2(x)
        y2 = self.conv4(self.conv3(y1))
        y3 = self.conv6(self.conv5(y2))

        return self.conv_out(torch.cat([y3, y2, y1, x1],dim=1))


class Pool_Conv_Block(nn.Module):
    def __init__(self,
                 channel:int):
        super().__init__()
        self.pool = Pool(padding=0)
        self.conv1 = Conv(in_channels=channel*2,
                          out_channels=channel,
                          kernel_size=1)
        self.conv2 = Conv(in_channels=channel*2,
                          out_channels=channel,
                          kernel_size=1)
        self.conv3 = Conv(in_channels=channel,
                          out_channels=channel,
                          kernel_size=3,
                          stride=2)

    def forward(self, x):
        return torch.cat([self.conv3(self.conv2(x)), self.conv1(self.pool(x))], dim=1)
    
class Stem(nn.Module):
    def __init__(self,in_channel, mid_channel, out_channel):
        super().__init__()
        self.conv1 = Conv(in_channels=in_channel,
                        out_channels=mid_channel,
                        kernel_size=3)
        self.conv2 = Conv(in_channels=mid_channel,
                          out_channels=out_channel,
                          kernel_size=3,
                          stride=2)
        self.conv3 = Conv(in_channels=out_channel,
                          out_channels=out_channel,
                          kernel_size=3)
        
    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))
    
@MODELS.register_module()
class YOLOV7Backbone(BaseModule):
    def __init__(self,
                stem_channels : Sequence[int] = [3, 32, 64],
                expand_list:Sequence[int] = [128, 256, 512, 1024, 1024],
                elan_list:Sequence[int] = [64, 128, 256, 256],
                num_out_stages:int = 3,
                init_cfg=dict(
                    type='Kaiming',
                    layer='Conv2d',
                    a=math.sqrt(5),
                    distribution='uniform',
                    mode='fan_in',
                    nonlinearity='ReLU')
                    ):
        super().__init__(init_cfg)
        self.num_out_stages = num_out_stages
        self.stem = Stem(*stem_channels)
        # self.conv1 = Conv(in_channels=stem_channels[-1],
        #                   out_channels=stem_channels[-1]*2,
        #                   kernel_size=3,
        #                   stride=2)
        self.stage1 = nn.Sequential(
            Conv(in_channels=stem_channels[-1],
                 out_channels=stem_channels[-1]*2,
                 kernel_size=3,
                 stride=2),
            Elan_BB(expand_list[0], elan_list[0], expand_list[1])     
        )
        #out_stages
        self.out_stages = nn.ModuleList()
        for idx in range(num_out_stages):
            self.out_stages.append(
                nn.Sequential(
                    Pool_Conv_Block(expand_list[idx]),
                    Elan_BB(expand_list[idx+1], elan_list[idx+1], expand_list[idx+2])
                ))

        # self.stage2 = nn.ModuleList(
        #     [Pool_Conv_Block(expand_list[0]),
        #     Elan_BB(expand_list[1], elan_list[1], expand_list[2]) ]       
        # )
        # self.stage3 = nn.ModuleList(
        #     [Pool_Conv_Block(expand_list[1]),
        #     Elan_BB(expand_list[2], elan_list[2], expand_list[3]) ]    
        # )
        # self.stage4 = nn.ModuleList([
        #     Pool_Conv_Block(expand_list[2]),
        #     Elan_BB(expand_list[3], elan_list[3], expand_list[4])]        
        # )

    def forward(self, x):
        outs = []
        x = self.stem(x)
        outs.append(self.stage1(x))
        for idx in range(self.num_out_stages):
            outs.append(self.out_stages[idx](outs[-1]))
        return outs[1:]



##test
        
# def test():
#     model = YOLOV7Backbone()
#     input = torch.rand(1,3,640,640)
#     output = model(input)
#     # print(model)
#     for out in output:
#         print(out.shape)

# test()