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

'''
An independent implementation of RegNet:
Designing Network Design Spaces, Ilija Radosavovic Raj Prateek Kosaraju Ross Girshick Kaiming He Piotr Dollar´,
Facebook AI Research (FAIR),
https://arxiv.org/pdf/2003.13678.pdf, https://github.com/facebookresearch/pycls
This implementation re-uses functions and classes from resnet.py
'''


import torch
import torch.nn as nn
import collections
from .utils import load_state_dict_from_url
from edgeai_torchmodelopt import xnn
from .resnet import conv1x1, conv3x3


__all__ = ['RegNet',
           'regnetx400mf', 'regnetx400mf_with_model_config',
           'regnetx800mf', 'regnetx800mf_with_model_config',
           'regnetx1p6gf', 'regnetx1p6gf_with_model_config',
           'regnetx3p2gf', 'regnetx3p2gf_with_model_config'
           ]


model_urls = {
    'regnetx400mf': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905967/RegNetX-400MF_dds_8gpu.pyth',
    'regnetx800mf': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906036/RegNetX-800MF_dds_8gpu.pyth',
    'regnetx1.6gf': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth',
    'regnetx3.2gf': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906139/RegNetX-3.2GF_dds_8gpu.pyth'
}


class RegNetBottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, group_width=None,
                 dilation=1, norm_layer=None):
        super(RegNetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        #
        groups = int(planes//group_width) if (group_width is not None) else 1
        width = int(planes//groups) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        a = conv1x1(inplanes, width)
        a_bn = norm_layer(width)
        a_relu = nn.ReLU(inplace=True)
        b = conv3x3(width, width, stride, groups, dilation)
        b_bn = norm_layer(width)
        b_relu = nn.ReLU(inplace=True)
        c = conv1x1(width, planes * self.expansion)
        c_bn = norm_layer(planes * self.expansion)
        self.f = torch.nn.Sequential(
            collections.OrderedDict([('a',a),('a_bn',a_bn),('a_relu',a_relu),
                 ('b',b),('b_bn',b_bn),('b_relu',b_relu),
                 ('c',c),('c_bn',c_bn)]))

        if downsample is not None:
            self.proj = downsample[0]
            self.bn = downsample[1]
            self.do_downsample = True
        else:
            self.do_downsample = False
        #
        self.add = xnn.layers.AddBlock()
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.f(x)
        if self.do_downsample:
            identity = self.bn(self.proj(x))

        out = self.add((out,identity))
        out = self.relu(out)

        return out


class RegNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 channels=(64,64,128,256,512), group_width=None, replace_stride_with_dilation=None,
                 norm_layer=None, input_channels=3, strides=None,
                 width_mult=1.0, fastdown=False, enable_fp16=False):
        super(RegNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        #
        self.group_width = group_width
        self._norm_layer = norm_layer
        self.num_classes = num_classes
        self.simple_stem = True
        self.enable_fp16 = enable_fp16

        self.inplanes = int(channels[0]*width_mult)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}")

        # strides of various layers
        strides = strides if (strides is not None) else (2,2,2,2,2)
        s0 = strides[0]
        s1 = strides[1]
        sf = 2 if (fastdown or self.simple_stem) else 1 # additional stride if fast down is true
        s2 = strides[2]
        s3 = strides[3]
        s4 = strides[4]

        if self.simple_stem:
            conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=3, stride=s0, padding=1, bias=False)
            bn1 = norm_layer(self.inplanes)
            relu = nn.ReLU(inplace=True)
            stem = [('conv',conv1), ('bn',bn1), ('relu1',relu)]
            if fastdown:
                maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                stem += [('maxpool', maxpool)]
            #
        else:
            conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=s0, padding=3, bias=False)
            bn1 = norm_layer(self.inplanes)
            relu = nn.ReLU(inplace=True)
            maxpool = nn.MaxPool2d(kernel_size=3, stride=s1, padding=1)
            stem = [('conv', conv1), ('bn', bn1), ('relu1', relu), ('maxpool', maxpool)]
        #
        stem = torch.nn.Sequential(collections.OrderedDict(stem))
        features = [('stem',stem)]

        layer1 = self._make_layer(block, int(channels[1]*width_mult), layers[0], stride=sf)
        layer2 = self._make_layer(block, int(channels[2]*width_mult), layers[1], stride=s2,
                                       dilate=replace_stride_with_dilation[0])
        layer3 = self._make_layer(block, int(channels[3]*width_mult), layers[2], stride=s3,
                                       dilate=replace_stride_with_dilation[1])
        layer4 = self._make_layer(block, int(channels[4]*width_mult), layers[3], stride=s4,
                                       dilate=replace_stride_with_dilation[2])

        features.append(('s1',layer1))
        features.append(('s2',layer2))
        features.append(('s3',layer3))
        features.append(('s4',layer4))
        self.features = torch.nn.Sequential(collections.OrderedDict(features))

        if self.num_classes:
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            flatten = torch.nn.Flatten(start_dim=1)
            fc = nn.Linear(int(channels[4]*width_mult) * block.expansion, num_classes)
            self.head = torch.nn.Sequential(collections.OrderedDict(
                [('avgpool',avgpool),('flatten',flatten),('fc',fc)]))

        xnn.utils.module_weights_init(self)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, RegNetBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, RegNetBottleneck):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = (
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        bx = block(self.inplanes, planes, stride, downsample, group_width=self.group_width,
                            dilation=previous_dilation, norm_layer=norm_layer)
        layers.append(('b1', bx))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            bx = block(self.inplanes, planes, group_width=self.group_width,
                                dilation=self.dilation, norm_layer=norm_layer)
            layers.append((f'b{i+1}', bx))
        #
        layers = torch.nn.Sequential(collections.OrderedDict(layers))
        return layers

    @xnn.utils.auto_fp16
    def forward(self, x):
        x = self.features(x)
        if self.num_classes:
            x = self.head(x)
        #
        return x

    # define a load weights fuinction in the module since the module is changed w.r.t. to torchvision
    # since we want to be able to laod the existing torchvision pretrained weights
    def load_weights(self, pretrained, change_names_dict=None, download_root=None):
        if change_names_dict is None:
            # the pretrained model provided by pycls and what is defined here differs slightly
            # note: that this change_names_dict  will take effect only if the direct load fails
            change_names_dict = {'^stem.': 'features.stem.',
                                 '^s1': 'features.s1',
                                 '^s2': 'features.s2',
                                 '^s3': 'features.s3',
                                 '^s4': 'features.s4'}
        #
        if pretrained is not None:
            xnn.utils.load_weights(self, pretrained, change_names_dict=change_names_dict,
                                   download_root=download_root, state_dict_name=['state_dict','model_state'])
        return self, change_names_dict


def _regnet(arch, block, layers, pretrained, progress, **kwargs):
    model = RegNet(block, layers, **kwargs)
    if pretrained is True:
        change_names_dict = kwargs.get('change_names_dict', None)
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_weights(state_dict, change_names_dict=change_names_dict)
    elif pretrained:
        change_names_dict = kwargs.get('change_names_dict', None)
        download_root = kwargs.get('download_root', None)
        model.load_weights(pretrained, change_names_dict=change_names_dict, download_root=download_root)
    return model


def regnetx400mf(pretrained=False, progress=True, **kwargs):
    r"""RegNet-400MF
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['channels'] = (32,32,64,160,384)
    kwargs['group_width'] = 16
    return _regnet('regnetx400mf', RegNetBottleneck, [1, 2, 7, 12],
                   pretrained, progress, **kwargs)


def regnetx800mf(pretrained=False, progress=True, **kwargs):
    r"""RegNet-800MF
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['channels'] = (32,64,128,288,672)
    kwargs['group_width'] = 16
    return _regnet('regnetx800mf', RegNetBottleneck, [1, 3, 7, 5],
                   pretrained, progress, **kwargs)


def regnetx1p6gf(pretrained=False, progress=True, **kwargs):
    r"""RegNet-1.6GF
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['channels'] = (32,72,168,408,912)
    kwargs['group_width'] = 24
    return _regnet('regnetx1p6gf', RegNetBottleneck, [2, 4, 10, 2],
                   pretrained, progress, **kwargs)


def regnetx3p2gf(pretrained=False, progress=True, **kwargs):
    r"""RegNet-3.2GF
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['channels'] = (32,96,192,432,1008)
    kwargs['group_width'] = 48
    return _regnet('regnetx3p2gf', RegNetBottleneck, [2, 6, 15, 2],
                   pretrained, progress, **kwargs)


###################################################
def get_config():
    model_config = xnn.utils.ConfigNode()
    model_config.input_channels = 3
    model_config.num_classes = 1000
    model_config.width_mult = 1.0
    model_config.strides = None
    model_config.fastdown = False
    model_config.enable_fp16 = False
    return model_config


def regnetx_base_with_model_config(model_config, model_func, pretrained=None):
    model_config = get_config().merge_from(model_config)
    model = model_func(input_channels=model_config.input_channels, strides=model_config.strides,
                     num_classes=model_config.num_classes, pretrained=pretrained,
                     width_mult=model_config.width_mult, fastdown=model_config.fastdown,
                     enable_fp16=model_config.enable_fp16)
    return model


def regnetx400mf_with_model_config(model_config, pretrained=None):
    return regnetx_base_with_model_config(model_config, regnetx400mf, pretrained=pretrained)


def regnetx800mf_with_model_config(model_config, pretrained=None):
    return regnetx_base_with_model_config(model_config, regnetx800mf, pretrained=pretrained)


def regnetx1p6gf_with_model_config(model_config, pretrained=None):
    return regnetx_base_with_model_config(model_config, regnetx1p6gf, pretrained=pretrained)


def regnetx3p2gf_with_model_config(model_config, pretrained=None):
    return regnetx_base_with_model_config(model_config, regnetx3p2gf, pretrained=pretrained)
