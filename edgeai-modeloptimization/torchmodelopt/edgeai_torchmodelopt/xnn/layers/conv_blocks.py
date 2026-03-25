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
from .layer_config import *
from . import functional
from .import common_blocks

def check_groups(in_planes, out_planes, groups, group_size, with_assert=True):
    assert groups is None or group_size is None, 'only one of groups or group_size must be specified'
    assert groups is not None or group_size is not None, 'atleast one of groups or group_size must be specified'
    groups = (in_planes//group_size) if groups is None else groups
    group_size = (in_planes//groups) if group_size is None else group_size
    if with_assert:
        assert in_planes%groups == 0, 'in_planes must be a multiple of groups'
        assert group_size != 1 or in_planes == out_planes, 'in DW layer channels must not change'
    #
    return groups, group_size

############################################################### 
def ConvLayer2d(in_planes, out_planes, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=False):
    """convolution with padding"""
    padding = padding if padding else ((kernel_size-1)//2)*dilation
    groups, group_size = check_groups(in_planes, out_planes, groups=groups, group_size=None)
    return DefaultConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)


def ConvDWLayer2d(in_planes, out_planes, kernel_size=None, stride=1, padding=None, dilation=1, groups_dw=None, group_size_dw=None, bias=False):
    """convolution with padding"""
    groups_dw = in_planes if (groups_dw is None and group_size_dw is None) else groups_dw
    groups_dw, group_size_dw = check_groups(in_planes, out_planes, groups=groups_dw, group_size=group_size_dw)
    return ConvLayer2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups_dw, bias=bias)
    

############################################################### 
def ConvNormAct2d(in_planes, out_planes, kernel_size=None, stride=1, padding=None, dilation=1, groups=1, bias=False, \
                normalization=DefaultNorm2d, activation=DefaultAct2d):
    """convolution with padding, BN, ReLU"""
    groups, group_size = check_groups(in_planes, out_planes, groups=groups, group_size=None)
    if type(kernel_size) in (list,tuple):
        padding = padding if padding else (((kernel_size[0]-1)//2)*dilation,((kernel_size[1]-1)//2)*dilation)
    else:
        padding = padding if padding else ((kernel_size-1)//2)*dilation

    if activation is True:
        activation = DefaultAct2d

    if normalization is True:
        normalization = DefaultNorm2d

    layers = [DefaultConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)]
    if normalization:
        layers.append(normalization(out_planes))

    if activation:
        layers.append(activation())

    layers = torch.nn.Sequential(*layers)
    return layers

    
def ConvDWNormAct2d(in_planes, out_planes, kernel_size=None, stride=1, padding=None, dilation=1, groups_dw=None, group_size_dw=None, bias=False,
                    normalization=DefaultNorm2d, activation=DefaultAct2d):
    """convolution with padding, BN, ReLU"""
    groups_dw = in_planes if (groups_dw is None and group_size_dw is None) else groups_dw
    groups_dw, group_size_dw = check_groups(in_planes, out_planes, groups=groups_dw, group_size=group_size_dw)
    return ConvNormAct2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups_dw, bias=bias, \
                     normalization=normalization, activation=activation)


###########################################################
def ConvDWSepNormAct2d(in_planes, out_planes, kernel_size=None, stride=1, padding=None,
                       dilation=1, groups=1, groups_dw=None, group_size_dw=None, bias=False, \
                       first_1x1=False, normalization=(DefaultNorm2d,DefaultNorm2d), activation=(DefaultAct2d,DefaultAct2d)):
    bias = bias if isinstance(bias, (list,tuple)) else (bias,bias)
    if first_1x1:
        layers = [ConvNormAct2d(in_planes, out_planes, kernel_size=1, groups=groups, bias=bias[0],
                      normalization=normalization[0], activation=activation[0]),
                  ConvDWNormAct2d(out_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups_dw=groups_dw, group_size_dw=group_size_dw, bias=bias[1],
                      normalization=normalization[1], activation=activation[1])]
    else:
        layers = [ConvDWNormAct2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups_dw=groups_dw, group_size_dw=group_size_dw, bias=bias[0],
                      normalization=normalization[0], activation=activation[0]),
                  ConvNormAct2d(in_planes, out_planes, kernel_size=1, bias=bias[1], groups=groups,
                      normalization=normalization[1], activation=activation[1])]

    layers = torch.nn.Sequential(*layers)
    return layers


###########################################################
def ConvDWTripletNormAct2d(in_planes, out_planes, kernel_size=None, stride=1,
                           padding=None, dilation=1, groups=1, groups_dw=None, group_size_dw=None, \
                           bias=False, intermediate_planes=None, expansion=1,
                           normalization=(DefaultNorm2d,DefaultNorm2d,DefaultNorm2d),
                           activation=(DefaultAct2d,DefaultAct2d,DefaultAct2d)):
    bias = bias if isinstance(bias, (list,tuple)) else (bias,bias,bias)

    if intermediate_planes is None:
        intermediate_planes = in_planes*expansion
    #
    groups_dw_, group_size_dw_ = check_groups(intermediate_planes, intermediate_planes, groups_dw, group_size_dw, with_assert=False)
    intermediate_planes = (intermediate_planes//groups_dw_) * groups_dw_

    layers = [ConvNormAct2d(in_planes, intermediate_planes, kernel_size=1, bias=bias[0], groups=groups,
                  normalization=normalization[0], activation=activation[0]),
              ConvDWNormAct2d(intermediate_planes, intermediate_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  groups_dw=groups_dw, group_size_dw=group_size_dw, bias=bias[1],
                  normalization=normalization[1], activation=activation[1]),
              ConvNormAct2d(intermediate_planes, out_planes, kernel_size=1, groups=groups, bias=bias[2],
                            normalization=normalization[2], activation=activation[2])]

    layers = torch.nn.Sequential(*layers)
    return layers


class ConvDWTripletRes2d(torch.nn.Module):
    def __init__(self, *args, with_residual=True, always_residual=False, activation_after_residual=True, **kwargs):
        super().__init__()

        in_planes = args[0]
        out_planes = args[1]

        # kernel_size = kwargs.get('kernel_size', None) or args[2]
        stride = kwargs.get('stride', 1) if len(args)<4 else args[3]
        bias = kwargs.get('bias', False)
        normalization = list(kwargs.get('normalization', [True,True,True]))
        activation = list(kwargs.get('activation', [True,True,True]))

        assert isinstance(normalization, (list, tuple)) and len(normalization) == 3, \
            'normalization must be a list/tuple with length 3'
        assert isinstance(activation, (list, tuple)) and len(activation) == 3, \
            'activation must be a list/tuple with length 3'

        is_shape_same = (in_planes == out_planes) and (stride == 1)
        self.use_residual = (with_residual and is_shape_same) or always_residual

        if self.use_residual:
            if activation_after_residual:
                # remove the last act fn from the list activation in kwargs
                # before creating the conv module ConvDWTripletNormAct2d
                activation_res, activation[-1] = activation[-1], False
                kwargs['activation'] = activation

        self.conv = ConvDWTripletNormAct2d(*args, **kwargs)

        if self.use_residual:
            # create residual connection if required
            if not is_shape_same:
                bias_last = bias[-1] if isinstance(bias, (list, tuple)) else bias
                self.res = ConvNormAct2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias_last,
                                normalization=normalization[-1], activation=activation[-1])
            else:
                self.res = None
            #
            # the residual addition module
            self.add = common_blocks.AddBlock()
            # create the last activation module if required
            if activation_after_residual:
                activation_res = DefaultAct2d if (activation_res is True) else activation_res
                self.act = activation_res() if activation_res else None
            else:
                self.act = None
            #
        #

    def forward(self, x):
        y = self.conv(x)
        if self.use_residual:
            x = self.res(x) if self.res is not None else x
            y = self.add((x,y))
            y = self.act(y) if self.act is not None else y
        #
        return y


class ConvDWTripletResAlways2d(ConvDWTripletRes2d):
    def __init__(self, *args, with_residual=True, activation_after_residual=True, **kwargs):
        super().__init__(*args, with_residual=with_residual, always_residual=True,
                         activation_after_residual=activation_after_residual, **kwargs)
