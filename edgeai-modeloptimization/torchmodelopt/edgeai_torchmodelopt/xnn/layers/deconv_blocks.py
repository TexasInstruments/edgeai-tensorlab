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
from .conv_blocks import *
from .layer_config import *
from .common_blocks import *

###############################################################
def DeConvLayer2d(in_planes, out_planes, kernel_size, stride=1, groups=1, dilation=1, padding=None, output_padding=None,
                  bias=False):
    """convolution with padding"""
    if (output_padding is None) and (padding is None):
        if kernel_size % 2 == 0:
            padding = (kernel_size - stride) // 2
            output_padding = 0
        else:
            padding = (kernel_size - stride + 1) // 2
            output_padding = 1

    return torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                    padding=padding,
                                    output_padding=output_padding, bias=bias, groups=groups)


def DeConvDWLayer2d(in_planes, out_planes, stride=1, dilation=1, kernel_size=None, padding=None, output_padding=None,
                    bias=False):
    """convolution with padding"""
    assert in_planes == out_planes, 'in DW layer channels must not change'
    return DeConvLayer2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                         groups=in_planes,
                         padding=padding, output_padding=output_padding, bias=bias)


###############################################################
def DeConvNormAct2d(in_planes, out_planes, kernel_size=None, stride=1, groups=1, dilation=1, padding=None,
                    output_padding=None, bias=False, \
                    normalization=DefaultNorm2d, activation=DefaultAct2d):
    """convolution with padding, BN, ReLU"""
    if (output_padding is None) and (padding is None):
        if kernel_size % 2 == 0:
            padding = (kernel_size - stride) // 2
            output_padding = 0
        else:
            padding = (kernel_size - stride + 1) // 2
            output_padding = 1

    if activation is True:
        activation = DefaultAct2d

    if normalization is True:
        normalization = DefaultNorm2d

    layers = [torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                       padding=padding,
                                       output_padding=output_padding, bias=bias, groups=groups)]
    if normalization:
        layers.append(normalization(out_planes))

    if activation:
        layers.append(activation(inplace=True))
    #
    layers = torch.nn.Sequential(*layers)
    return layers


def DeConvDWNormAct2d(in_planes, out_planes, stride=1, kernel_size=None, dilation=1, padding=None, output_padding=None,
                      bias=False,
                      normalization=DefaultNorm2d, activation=DefaultAct2d):
    """convolution with padding, BN, ReLU"""
    assert in_planes == out_planes, 'in DW layer channels must not change'
    return DeConvNormAct2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                           padding=padding, output_padding=output_padding,
                           bias=bias, groups=in_planes, normalization=normalization, activation=activation)


###########################################################
def DeConvDWSepNormAct2d(in_planes, out_planes, stride=1, kernel_size=None, groups=1, dilation=1, bias=False, \
                         first_1x1=False, normalization=(DefaultNorm2d, DefaultNorm2d),
                         activation=(DefaultAct2d, DefaultAct2d)):
    if first_1x1:
        layers = [
            ConvNormAct2d(in_planes, out_planes, kernel_size=1, groups=groups, bias=bias,
                          normalization=normalization[0], activation=activation[0]),
            DeConvDWNormAct2d(out_planes, out_planes, stride=stride, kernel_size=kernel_size, dilation=dilation,
                              bias=bias,
                              normalization=normalization[1], activation=activation[1])]
    else:
        layers = [DeConvDWNormAct2d(in_planes, in_planes, stride=stride, kernel_size=kernel_size, dilation=dilation,
                                    bias=bias,
                                    normalization=normalization[0], activation=activation[0]),
                  ConvNormAct2d(in_planes, out_planes, groups=groups, kernel_size=1, bias=bias,
                                normalization=normalization[1], activation=activation[1])]

    layers = torch.nn.Sequential(*layers)
    return layers

