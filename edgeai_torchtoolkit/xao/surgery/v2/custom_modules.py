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

import torch
from torch import nn , Tensor   


# Squeeze and excitation module with relu and hardsigmoid as activation function
class SEModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequence=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels= 32, out_channels= 16,kernel_size=1,),
            nn.Hardsigmoid()
        )
    
    def forward(self,x):
        return torch.mul(self.sequence(x),x)


# Squeeze and excitation module with silu and sigmoid as activation function
class SEModule1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequence=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels= 32, out_channels= 16,kernel_size=1,),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        return torch.mul(self.sequence(x),x)


# Wrapper module for modules in nn package
class InstaModule(nn.Module):
    def __init__(self,preDefinedLayer:nn.Module) -> None:
        super().__init__()
        self.model=preDefinedLayer

    def forward(self,x):
        return self.model(x)


# focus module for segmentation models
class Focus(nn.Module):
    def forward(self,x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return x


# a typical convulation module to be used as replacement
class ConvBNRModule(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding) -> None:
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.bn=nn.BatchNorm2d(out_channels)
        self.act=nn.ReLU()
    
    def forward(self,x,*args):
        return self.act(self.bn(self.conv(x)))


class ReplaceBatchNorm2d(nn.Module):
        def __init__(self, num_features) -> None:
            super().__init__()
            self.bn=nn.BatchNorm2d(num_features=num_features)
        def forward(self,x:Tensor):
            out= x.permute(0,3,1,2)
            out= self.bn(out)
            return out.permute(0,2,3,1)


class ReplacementCNBlock(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNRModule(dim,dim,kernel_size=3,stride=1,padding=1),
            ConvBNRModule(dim,dim,kernel_size=5,stride=1,padding=2),
            ConvBNRModule(dim,4*dim,kernel_size=1,stride=1,padding=0),
            nn.Conv2d(4*dim,dim,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(dim)
        )
    
    def forward(self,x):
        result = self.block(x)
        result += x
        return result