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
from .utils import *
from edgeai_torchmodelopt import xnn

###################################################
__all__ = ['MobileNetV1Base', 'MobileNetV1', 'mobilenet_v1', 'get_config']


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
    model_config.classifier_type = torch.nn.Linear
    model_config.en_make_divisible_by8 = True
    model_config.enable_fp16 = False
    return model_config

model_urls = {
    'mobilenet_v1': None,
}


class MobileNetV1Base(torch.nn.Module):
    def __init__(self, BlockBuilder, model_config):
        """
        MobileNet V1 main class
        """
        super().__init__()
        self.model_config = model_config
        self.num_classes = self.model_config.num_classes
        self.enable_fp16 = model_config.enable_fp16

        # strides of various layers
        strides = model_config.strides if (model_config.strides is not None) else (2,2,2,2,2)
        s0 = strides[0]
        s1 = strides[1]
        s2 = strides[2]
        s3 = strides[3]
        s4 = strides[4]

        if self.model_config.layer_setting is None:
            self.model_config.layer_setting = [
                # t,  c,  n,  s
                [1,  32,  1, s0],
                [1,  64,  1,  1],
                [1, 128,  2, s1],
                [1, 256,  2, s2],
                [1, 512,  6, s3],
                [1,1024,  2, s4],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(self.model_config.layer_setting) == 0 or len(self.model_config.layer_setting[0]) != 4:
            raise ValueError(f"inverted_residual_setting should be non-empty or a 4-element list, got {self.model_config.layer_setting}")

        # some params
        activation = self.model_config.activation
        width_mult = self.model_config.width_mult
        kernel_size = self.model_config.kernel_size

        # building first layer
        output_channels = int(self.model_config.layer_setting[0][1] * width_mult)
        output_channels = xnn.utils.make_divisible_by8(output_channels) if model_config.en_make_divisible_by8 else output_channels
        features = [xnn.layers.ConvNormAct2d(3, output_channels, kernel_size=kernel_size, stride=s0, activation=activation)]
        channels = output_channels

        # building inverted residual blocks
        for t, c, n, s in self.model_config.layer_setting[1:]:
            output_channels = xnn.utils.make_divisible_by8(c * width_mult) if model_config.en_make_divisible_by8 else int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                block = BlockBuilder(channels, output_channels, stride=stride, kernel_size=kernel_size, activation=(activation,activation))
                features.append(block)
                channels = output_channels
            #
        #

        # building classifier
        if self.model_config.num_classes != None:
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.2) if self.model_config.dropout else xnn.layers.BypassBlock(),
                model_config.classifier_type(channels, self.num_classes),
            )
        #

        # make it sequential
        self.features = torch.nn.Sequential(*features)

        # weights init
        xnn.utils.module_weights_init(self)

    @xnn.utils.auto_fp16
    def forward(self, x):
        x = self.features(x)
        if self.num_classes is not None:
            xnn.utils.print_once('=> feature size is: ', x.size())
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1))
            #xnn.utils.print_once('=> size after pool2d: ', x.size())
            x = torch.flatten(x, 1)
            #xnn.utils.print_once('=> size after flatten: ', x.size())
            x = self.classifier(x)
            #xnn.utils.print_once('=> size after classifier: ', x.size())
        #
        return x


class MobileNetV1(MobileNetV1Base):
    def __init__(self, **kwargs):
        model_config = get_config()
        if 'model_config' in list(kwargs.keys()):
            model_config = model_config.merge_from(kwargs['model_config'])
        super().__init__(xnn.layers.ConvDWSepNormAct2d, model_config)


#######################################################################
def mobilenet_v1(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV1

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV1(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v1'], progress=progress)
        model.load_state_dict(state_dict)
    return model

