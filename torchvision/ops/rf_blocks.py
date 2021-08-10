#################################################################################
# Copyright (c) 2018-2021, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################


from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from typing import Tuple, List, Dict, Optional, Union

from ..import xnn

from .feature_pyramid_network import ExtraFPNBlock, LastLevelP6P7, LastLevelMaxPool


# An simplified form of BiFPN mentioned in:
# https://arxiv.org/abs/1911.09070

class BiFPN(nn.Module):
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        conv_cfg: Union[dict,None] = None,
        num_outputs=6,
        start_level=3,
        num_blocks=1
    ):
        super().__init__()
        self.num_outputs = num_outputs
        self.intermediate_channels = out_channels
        self.out_channels = out_channels
        self.start_level = start_level
        self.num_backbone_convs = len(in_channels_list)
        self.extra_levels = num_outputs - self.num_backbone_convs

        assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks

        # align channels of input (e.g., RetinaNet)
        self.in_channels = []
        self.in_convs = nn.ModuleList()
        for i in range(num_outputs):
            if i < self.num_backbone_convs:
                in_ch = in_channels_list[i]
                in_conv = xnn.layers.ConvNormWrapper2d(in_ch, out_channels, kernel_size=1, conv_cfg=conv_cfg)
            else:
                in_ch = out_channels
                in_conv = torch.nn.Identity()
            #
            self.in_channels.append(in_ch)
            self.in_convs.append(in_conv)
        #

        blocks = []
        for block_id in range(num_blocks):
            last_in_channels = [self.intermediate_channels for _ in range(self.num_outputs)] if block_id>0 else self.in_channels
            up_only = (block_id == (num_blocks-1)) and (out_channels != self.intermediate_channels)
            bi_fpn = BiFPNBlock(block_id=block_id, up_only=up_only, in_channels=last_in_channels, out_channels=out_channels,
                                    num_outputs=self.num_outputs, start_level=start_level, conv_cfg=conv_cfg)
            blocks.append(bi_fpn)
        #
        self.bifpn_blocks = nn.ModuleList(blocks)
        # use leaky relu style initialization as the output is signed (no ReLU at output)
        xnn.utils.module_weights_init(self, weight_init='uniform', mode='fan_in', nonlinearity='leaky_relu', a=1)

    def forward(self, inputs_dict):
        inputs = list(inputs_dict.values())
        names = list(inputs_dict.keys())

        if self.extra_blocks is not None:
            inputs, names = self.extra_blocks(inputs, inputs, names)

        # in convs
        ins = []
        for i in range(self.num_outputs):
            ins.append(self.in_convs[i](inputs[i]))
        #
        assert len(ins) == len(self.in_channels)

        results = ins
        for block in self.bifpn_blocks:
            results, names = block(results)

        outputs = OrderedDict([(k, v) for k, v in zip(names, results)])

        return outputs


class BiFPNBlock(nn.Module):
    def __init__(self, block_id=None, up_only=False, in_channels=None, out_channels=None,
                 num_outputs=5, start_level=3, conv_cfg=None, upsample_cfg=None):
        super(BiFPNBlock, self).__init__()
        assert isinstance(in_channels, (list,tuple))

        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outputs = num_outputs
        self.start_level = start_level
        self.up_only = up_only
        self.block_id = block_id

        assert block_id is not None, f'block_id must be valid: {block_id}'

        # Use act only if conv already has act
        ActType = nn.Identity
        DownsampleType = nn.MaxPool2d
        UpsampleType = xnn.layers.ResizeWith
        upsample_cfg = dict(scale_factor=2, mode='nearest') if upsample_cfg is None else upsample_cfg

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_acts = nn.ModuleList()
        self.up_adds = nn.ModuleList()
        if not up_only:
            self.downs = nn.ModuleList()
            self.down_convs = nn.ModuleList()
            self.down_acts = nn.ModuleList()
            self.down_adds1 = nn.ModuleList()
            self.down_adds2 = nn.ModuleList()
        #
        for i in range(self.num_outputs-1):
            # up modules
            if not up_only:
                up = UpsampleType(**upsample_cfg)
                up_conv = xnn.layers.ConvNormWrapper2d(out_channels,
                        out_channels, 3, padding=1,
                        conv_cfg=conv_cfg)
                up_act = ActType()
                self.ups.append(up)
                self.up_convs.append(up_conv)
                self.up_acts.append(up_act)
                self.up_adds.append(xnn.layers.AddBlock())
            #
            # down modules
            down = DownsampleType(kernel_size=3, stride=2, padding=1)
            down_conv = xnn.layers.ConvNormWrapper2d(out_channels,
                out_channels, 3, padding=1,
                conv_cfg=conv_cfg)
            down_act = ActType()
            self.downs.append(down)
            self.down_convs.append(down_conv)
            self.down_acts.append(down_act)
            self.down_adds1.append(xnn.layers.AddBlock())
            self.down_adds2.append(xnn.layers.AddBlock())

    def forward(self, inputs):
        ins = inputs
        # up convs
        ups = [None] * self.num_outputs
        ups[-1] = ins[-1]
        for i in range(self.num_outputs-2, -1, -1):
            add_block = self.up_adds[i]
            ups[i] = self.up_convs[i](self.up_acts[i](
                    add_block((ins[i], self.ups[i](ups[i+1])))
            ))
        #
        if self.up_only:
            return tuple(ups)
        #
        # down convs
        outs = [None] * self.num_outputs
        outs[0] = ups[0]
        for i in range(0, self.num_outputs-1):
            add_block1 = self.down_adds1[i]
            res = add_block1((ins[i+1], ups[i+1])) if (ins[i+1] is not ups[i+1]) else ins[i+1]
            add_block2 = self.down_adds2[i]
            outs[i+1] = self.down_convs[i](self.down_acts[i](
                add_block2((res,self.downs[i](outs[i])))
            ))

        names = [f'p{self.start_level+n}' for n in range(self.num_outputs)]
        return tuple(outs), names


# TODO: implement path aggregation network (PAN) or its Concat variant as mentioned in YOLOV4:
# https://arxiv.org/pdf/2004.10934.pdf
# https://arxiv.org/pdf/2011.08036.pdf
# https://arxiv.org/abs/1803.01534
