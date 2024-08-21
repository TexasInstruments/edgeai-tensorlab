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


# Note: Some of the modules from this file are copied to surgery v3. 
#       So if changes are made in them, those must be done there as well

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
    
    
# focus module for segmentation models
class OptimizedFocus(nn.Module):
    def forward(self,x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = x.permute(0, 3, 1, 2)
        return x


# a typical convulation module to be used as replacement
class ConvBNRModule(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,groups=1) -> None:
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding, groups=groups)
        self.bn=nn.BatchNorm2d(out_channels)
        self.act=nn.ReLU()
    
    def forward(self,x,*args):
        return self.act(self.bn(self.conv(x)))


class Permute(torch.nn.Module):
    def __init__(self,shape:list|tuple, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.shape = shape
    def forward(self,x):
        return torch.permute(x,self.shape)


class ReplaceBatchNorm(nn.Module):
    def __init__(self,num_features ,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_features = num_features
        self.layer = None
        self.permute1= None
        self.permute2= None
        self.module_list = None
    
    # 
    def forward(self,x:torch.Tensor):
        if self.module_list:
            return self.module_list(x)
        if len(x.shape) ==3:
            # x= x.permute(0,2,1)
            # x = self.layer(x)
            # return x.permute(0,2,1)
            self.permute1 = self.permute1 or Permute([0,2,1])
            self.layer =self.layer or  nn.BatchNorm1d(self.num_features)
            self.permute2 = self.permute2 or Permute([0,2,1])
            self.module_list= nn.Sequential(
                self.permute1,
                self.layer,
                self.permute2
            )
        elif len(x.shape) ==4:
            # x = x.permute(0,3,1,2)
            # x = self.layer(x)
            # return x.permute(0,2,3,1)
            self.permute1 = self.permute1 or Permute([0,3,1,2])
            self.layer =self.layer or  nn.BatchNorm2d(self.num_features)
            self.permute2 = self.permute2 or Permute([0,2,3,1])
            self.module_list= nn.Sequential(
                self.permute1,
                self.layer,
                self.permute2
            )
        else:
            self.module_list = nn.Identity() 
        return self.module_list(x)


class ReplacementCNBlock(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNRModule(dim,dim,kernel_size=5,stride=1,padding=2, groups=dim), # depthwise conv layer
            ConvBNRModule(dim,dim,kernel_size=3,stride=1,padding=1, groups=dim), # depthwise conv layer 
            ConvBNRModule(dim,4*dim,kernel_size=1,stride=1,padding=0),
            nn.Conv2d(4*dim,dim,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(dim)
        )
    
    def forward(self,x):
        result = self.block(x)
        result += x
        return result