'''
# Derived from: https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv2.py (modified to make it mobilenetv1)

==============================================================================
Texas Instruments (C) 2018-2019
All Rights Reserved

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==============================================================================
Some parts of the code are borrowed from: https://github.com/pytorch/vision
with the following license:

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import torch
import logging
import numpy  as np
import warnings
from mmcv.runner import BaseModule

from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES
from edgeai_xvision import xnn

###################################################
__all__ = ['MobileNetV1LiteBase', 'MobileNetV1Lite', 'mobilenet_v1_lite']


###################################################
class ModelConfig(xnn.utils.ConfigNode):
    def __init__(self):
        super().__init__()
        self.input_channels = 3
        self.num_classes = None
        self.width_mult = 1.
        self.expand_ratio = 6
        self.strides = (2,2,2,2,2)
        self.activation = xnn.layers.DefaultAct2d
        self.use_blocks = False
        self.kernel_size = 3
        self.dropout = False
        self.linear_dw = False
        self.layer_setting = None
        self.out_indices = None
        self.shortcut_channels = (32,128,256,512,1024)
        self.frozen_stages = 0
        self.extra_channels = None
        self.act_cfg = None

    @property
    def shortcut_strides(self):
        encoder_stride = np.prod(self.strides)
        s_strides = (2,4,8,16,encoder_stride)
        return s_strides

def get_config():
    return ModelConfig()

model_urls = {
    'mobilenet_v1': None,
}


class MobileNetV1LiteBase(BaseModule):
    def __init__(self, BlockBuilder, model_config, pretrained=None, init_cfg=None):
        """
        MobileNet V1 main class
        """
        super().__init__(init_cfg)

        self.model_config = model_config
        self.num_classes = self.model_config.num_classes

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        # strides of various layers
        s0 = model_config.strides[0]
        s1 = model_config.strides[1]
        s2 = model_config.strides[2]
        s3 = model_config.strides[3]
        s4 = model_config.strides[4]

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
        output_channels = xnn.utils.make_divisible_by8(self.model_config.layer_setting[0][1] * width_mult)
        features = [xnn.layers.ConvNormAct2d(3, output_channels, kernel_size=kernel_size, stride=s0, activation=activation)]
        channels = output_channels

        # building inverted residual blocks
        for t, c, n, s in self.model_config.layer_setting[1:]:
            output_channels = xnn.utils.make_divisible_by8(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                block = BlockBuilder(channels, output_channels, stride=stride, kernel_size=kernel_size, activation=(activation,activation))
                features.append(block)
                channels = output_channels
            #
        #

        # building classifier
        if self.model_config.num_classes is not None:
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.2) if self.model_config.dropout else xnn.layers.BypassBlock(),
                torch.nn.Linear(channels, self.num_classes),
            )
        #

        # make it sequential
        self.features = torch.nn.Sequential(*features)

    def forward(self, x):
        x = self.features(x)
        xnn.utils.print_once('=> feature size is: ', x.size())
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@BACKBONES.register_module
class MobileNetV1Lite(MobileNetV1LiteBase):
    def __init__(self, pretrained=None, init_cfg=None, **kwargs):
        model_config = get_config()
        for key, value in kwargs.items():
            if key == 'model_config':
                model_config = model_config.merge_from(value)
            elif key in ('out_indices', 'strides', 'extra_channels', 'frozen_stages', 'act_cfg'):
                setattr(model_config, key, value)
        #
        super().__init__(xnn.layers.ConvDWSepNormAct2d, model_config, pretrained=pretrained, init_cfg=init_cfg)

        self.extra = self._make_extra_layers(1024, self.model_config.extra_channels) \
            if self.model_config.extra_channels else None

        # weights init
        xnn.utils.module_weights_init(self)

    # def init_weights(self, pretrained=None):
    #     if pretrained is not None:
    #         assert isinstance(pretrained, str), f'Make sure that the pretrained is correct. Got: {pretrained}'
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, strict=False, logger=logger)
    #     else:
    #         warnings.warn('No pretrained is provided.')

    def forward(self, x):
        if self.num_classes is not None:
            x = super().forward(x)
            return x
        else:
            in_shape = x.shape
            x_list = []
            for layer in self.features:
                x = layer(x)
                x_list.append(x)
            #
            out = []
            shortcut_strides = self.model_config.shortcut_strides
            for s_stride, short_chan in zip(shortcut_strides, self.model_config.shortcut_channels):
                shape_s = xnn.utils.get_shape_with_stride(in_shape, s_stride)
                shape_s[1] = short_chan
                # do not want this to be traced by jit
                shape_s = [int(s) for s in shape_s]
                x_s = xnn.utils.get_blob_from_list(x_list, shape_s)
                out.append(x_s)

            if self.model_config.out_indices is not None:
                selected_out = []
                for i, o in enumerate(out):
                    if i in self.model_config.out_indices:
                        selected_out.append(o)
                #
            else:
                selected_out = out
            #
            if self.extra:
                for layer in self.extra:
                    x = layer(x)
                    selected_out.append(x)
                #
            #
            if self.model_config.frozen_stages>0:
                selected_out = [o.detach() for o in selected_out]
            #
            return selected_out
        #


    def _make_extra_layers(self, inplanes, outplanes, kernel_size=3):
        act_cfg = self.model_config.act_cfg
        act_dw = (act_cfg is None) or ('act_dw' not in act_cfg) or act_cfg['act_dw']
        extra_layers = []
        for i, out_ch in enumerate(outplanes):
            activation = (act_dw, True)
            layer = xnn.layers.ConvDWSepNormAct2d(inplanes, out_ch, stride=2, kernel_size=kernel_size, activation=activation)
            extra_layers.append(layer)
            inplanes = out_ch
        #
        return torch.nn.Sequential(*extra_layers)


#######################################################################
def mobilenet_v1_lite(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV1
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV1(**kwargs)
    if pretrained:
        state_dict = xnn.utils.load_state_dict_from_url(model_urls['mobilenet_v1'], progress=progress)
        model.load_state_dict(state_dict)
    return model
