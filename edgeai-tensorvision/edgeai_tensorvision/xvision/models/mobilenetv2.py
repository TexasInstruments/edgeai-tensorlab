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
# Some parts of the code are borrowed from: https://github.com/pytorch/vision
# with the following license:
#
# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
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
from torch import nn, Tensor
from typing import Any, Callable, Dict, List, Optional, Sequence

from .utils import *
from edgeai_torchmodelopt import xnn

###################################################
__all__ = ['MobileNetV2Base', 'MobileNetV2',
           'MobileNetV2TVBase', 'MobileNetV2TV',
           'mobilenet_v2', 'mobilenet_v2_tv', 'get_config']


###################################################
def get_config():
    model_config = xnn.utils.ConfigNode()
    model_config.input_channels = 3
    model_config.num_classes = 1000
    model_config.width_mult = 1.
    model_config.expand_ratio = 6
    model_config.strides = None #(2,2,2,2,2)
    model_config.activation = xnn.layers.DefaultAct2d
    model_config.use_blocks = False
    model_config.kernel_size = 3
    model_config.dropout = False
    model_config.linear_dw = False
    model_config.layer_setting = None
    model_config.fastdown = False
    model_config.enable_fp16 = False
    return model_config

model_urls = {
    'mobilenet_v2_tv': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(torch.nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, activation, kernel_size, linear_dw):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        activation_dw = (False if linear_dw else activation)

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(xnn.layers.ConvNormAct2d(inp, hidden_dim, kernel_size=1, activation=activation))
        #
        layers.extend([
            # dw
            xnn.layers.ConvNormAct2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim, activation=activation_dw),
            # pw-linear
            torch.nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            xnn.layers.DefaultNorm2d(oup),
        ])

        if linear_dw:
            layers.append(activation(inplace=True))


        self.conv = torch.nn.Sequential(*layers)

        if self.use_res_connect:
            self.add = xnn.layers.AddBlock(signed=True)


    def forward(self, x):
        if self.use_res_connect:
            return self.add((x, self.conv(x)))
        else:
            return self.conv(x)



class MobileNetV2Base(torch.nn.Module):
    def __init__(self, BlockBuilder, model_config):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super().__init__()
        self.model_config = model_config
        self.num_classes = self.model_config.num_classes
        self.enable_fp16 = model_config.enable_fp16

        # strides of various layers
        strides = model_config.strides if (model_config.strides is not None) else (2,2,2,2,2)
        s0 = strides[0]
        sf = 2 if model_config.fastdown else 1 # extra stride if fastdown
        s1 = strides[1]
        s2 = strides[2]
        s3 = strides[3]
        s4 = strides[4]

        if self.model_config.layer_setting is None:
            ex = self.model_config.expand_ratio
            self.model_config.layer_setting = [
                # t, c,  n, s
                [1,  32,  1, s0],
                [1,  16,  1, sf],
                [ex, 24,  2, s1],
                [ex, 32,  3, s2],
                [ex, 64,  4, s3],
                [ex, 96,  3,  1],
                [ex, 160, 3, s4],
                [ex, 320, 1,  1],
                [1, 1280, 1,  1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(self.model_config.layer_setting) == 0 or len(self.model_config.layer_setting[0]) != 4:
            raise ValueError(f"inverted_residual_setting should be non-empty or a 4-element list, got {self.model_config.layer_setting}")

        # some params
        activation = self.model_config.activation
        width_mult = self.model_config.width_mult
        linear_dw = self.model_config.linear_dw
        kernel_size = self.model_config.kernel_size

        # building first layer
        output_channels = xnn.utils.make_divisible_by8(self.model_config.layer_setting[0][1] * width_mult)
        features = [xnn.layers.ConvNormAct2d(model_config.input_channels, output_channels, kernel_size=kernel_size, stride=s0, activation=activation)]
        channels = output_channels

        # building inverted residual blocks
        for t, c, n, s in self.model_config.layer_setting[1:-1]:
            output_channels = xnn.utils.make_divisible_by8(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                block = BlockBuilder(channels, output_channels, stride=stride, kernel_size=kernel_size, activation=activation, expand_ratio=t, linear_dw=linear_dw)
                features.append(block)
                channels = output_channels
            #
        #

        # building classifier
        if self.model_config.num_classes != None:
            output_channels = xnn.utils.make_divisible_by8(self.model_config.layer_setting[-1][1] * width_mult)
            features.append(xnn.layers.ConvNormAct2d(channels, output_channels, kernel_size=1, activation=activation))
            channels = output_channels 
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.2) if self.model_config.dropout else xnn.layers.BypassBlock(),
                torch.nn.Linear(channels, self.num_classes),
            )
        #

        # make it sequential
        self.features = torch.nn.Sequential(*features)

        # weights init
        xnn.utils.module_weights_init(self)


    def _forward_impl(self, x):
        x = self.features(x)
        if self.num_classes is not None:
            xnn.utils.print_once('=> feature size is: ', x.size())
            x = torch.nn.functional.adaptive_avg_pool2d(x,(1,1))
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        #
        return x


    @xnn.utils.auto_fp16
    def forward(self, x):
        return self._forward_impl(x)


class MobileNetV2(MobileNetV2Base):
    def __init__(self, **kwargs):
        model_config = get_config()
        if 'model_config' in list(kwargs.keys()):
            model_config = model_config.merge_from(kwargs['model_config'])
        super().__init__(InvertedResidual, model_config)


#######################################################################
def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2_tv'], progress=progress)
        model.load_state_dict(state_dict)
    return model


#######################################################################
# just another name - TV is used indicating this is from torchvision
MobileNetV2TVBase = MobileNetV2Base
MobileNetV2TV = MobileNetV2
mobilenet_v2_tv = mobilenet_v2