#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
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

# blocks to improve the receptive field. eg. ASPP, LargeHead

import torch
import torch.nn.functional as F
from .conv_blocks import *
from .common_blocks import *


######################################################
class ASPPBlock(torch.nn.Module):
    def __init__(self, in_chs, aspp_chs, out_chs, dilation=(6, 12, 18), groups=1, avg_pool=True, activation=DefaultAct2d):
        super().__init__()

        self.aspp_chs = aspp_chs
        self.avg_pool = avg_pool
        self.last_chns = aspp_chs * (4 + (1 if self.avg_pool else 0))

        self.aspp_in = ConvNormAct2d(in_chs, aspp_chs, kernel_size=1, activation=None)

        if self.avg_pool:
            self.gave_pool = torch.nn.Sequential(activation(inplace=False), torch.nn.AdaptiveAvgPool2d((1, 1)),
                                           torch.nn.Conv2d(aspp_chs, aspp_chs, kernel_size=1, bias=False),
                                           activation(inplace=True))
        #

        self.conv1x1 = ConvNormAct2d(aspp_chs, aspp_chs, kernel_size=1, groups=groups, activation=activation)
        self.aspp_bra1 = ConvNormAct2d(aspp_chs, aspp_chs, kernel_size=3, groups=groups, dilation=dilation[0], activation=activation)
        self.aspp_bra2 = ConvNormAct2d(aspp_chs, aspp_chs, kernel_size=3, groups=groups, dilation=dilation[1], activation=activation)
        self.aspp_bra3 = ConvNormAct2d(aspp_chs, aspp_chs, kernel_size=3, groups=groups, dilation=dilation[2], activation=activation)

        self.dropout = torch.nn.Dropout2d(p=0.2, inplace=True)
        self.aspp_out = ConvNormAct2d(self.last_chns, out_chs, kernel_size=1, activation=activation)
        self.cat = CatBlock()

    def forward(self, x):
        x0 = self.aspp_in(x)

        x1 = self.conv1x1(x0)
        b1 = self.aspp_bra1(x0)
        b2 = self.aspp_bra2(x0)
        b3 = self.aspp_bra3(x0)

        if self.avg_pool:
            xavg = torch.nn.functional.interpolate(self.gave_pool(x0), size=x0.shape[2:], mode='bilinear')
            branches = [xavg, x1, b1, b2, b3]
        else:
            branches = [x1, b1, b2, b3]
        #

        cat = self.cat(branches)
        cat = self.dropout(cat)
        out = self.aspp_out(cat)

        return out


######################################################
# this is called a lite block because the dilated convolutions use
# ConvDWNormAct2d instead of ConvDWSepNormAct2d
class DWASPPLiteBlock(torch.nn.Module):
    def __init__(self, in_chs, aspp_chs, out_chs, dilation=(6, 12, 18), groups=1, group_size_dw=None, avg_pool=False,
                 activation=DefaultAct2d, linear_dw=False):
        super().__init__()

        self.aspp_chs = aspp_chs
        self.avg_pool = avg_pool
        self.last_chns = aspp_chs * (4 + (1 if self.avg_pool else 0))

        if self.avg_pool:
            self.gave_pool = torch.nn.Sequential(activation(inplace=False), torch.nn.AdaptiveAvgPool2d((1, 1)),
                                           torch.nn.Conv2d(in_chs, aspp_chs, kernel_size=1), activation(inplace=True))
        #

        self.conv1x1 = ConvNormAct2d(in_chs, aspp_chs, kernel_size=1, activation=activation)
        normalizations_dw = ((not linear_dw), True)
        activations_dw = (False if linear_dw else activation, activation)
        self.aspp_bra1 = ConvDWSepNormAct2d(in_chs, aspp_chs, kernel_size=3, dilation=dilation[0],
                                            normalization=normalizations_dw, activation=activations_dw,
                                            groups=groups, group_size_dw=group_size_dw)
        self.aspp_bra2 = ConvDWSepNormAct2d(in_chs, aspp_chs, kernel_size=3, dilation=dilation[1],
                                            normalization=normalizations_dw, activation=activations_dw,
                                            groups=groups, group_size_dw=group_size_dw)
        self.aspp_bra3 = ConvDWSepNormAct2d(in_chs, aspp_chs, kernel_size=3, dilation=dilation[2],
                                            normalization=normalizations_dw, activation=activations_dw,
                                            groups=groups, group_size_dw=group_size_dw)

        self.dropout = torch.nn.Dropout2d(p=0.2, inplace=True)   
        self.aspp_out = ConvNormAct2d(self.last_chns, out_chs, kernel_size=1, groups=1, activation=activation)
        self.cat = CatBlock()

    def forward(self, x):
        x1 = self.conv1x1(x)
        b1 = self.aspp_bra1(x)
        b2 = self.aspp_bra2(x)
        b3 = self.aspp_bra3(x)

        if self.avg_pool:
            xavg = F.interpolate(self.gave_pool(self.aspp_in(x)), size=x.shape[2:], mode='bilinear')
            branches = [xavg, x1, b1, b2, b3]
        else:
            branches = [x1, b1, b2, b3]
        #

        cat = self.cat(branches)
        cat = self.dropout(cat)
        out = self.aspp_out(cat)
        return out
#


