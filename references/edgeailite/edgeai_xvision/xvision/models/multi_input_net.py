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

import copy
import torch
from edgeai_torchmodelopt import xnn
from .mobilenetv2 import MobileNetV2TV
from .resnet import resnet50_with_model_config
from .regnet import regnetx400mf_with_model_config, regnetx800mf_with_model_config, \
                    regnetx1p6gf_with_model_config, regnetx3p2gf_with_model_config

try: from .mobilenetv2_ericsun_internal import *
except: pass

try: from .mobilenetv2_internal import *
except: pass

__all__ = ['MultiInputNet', 'mobilenet_v2_tv_mi4', 'mobilenet_v2_tv_gws_mi4', 'mobilenet_v2_ericsun_mi4',
           'MobileNetV2TVMI4', 'MobileNetV2TVNV12MI4', 'ResNet50MI4',
           'RegNetX400MFMI4', 'RegNetX800MFMI4', 'RegNetX1p6GFMI4', 'RegNetX3p2GFMI4']


###################################################
def get_config():
    model_config = xnn.utils.ConfigNode()
    model_config.num_inputs = 1
    model_config.input_channels = 3
    model_config.num_classes = 1000
    model_config.fuse_channels = 0
    model_config.intermediate_outputs = False
    model_config.num_input_blocks = 0
    model_config.shared_weights = False
    model_config.fuse_stride = 1
    return model_config


###################################################
class MultiInputNet(torch.nn.Module):
    def __init__(self, Model, model_config, pretrained=None):
        model_config = get_config().merge_from(model_config)
        super().__init__()

        self.num_classes = model_config.num_classes
        self.num_inputs = len(model_config.input_channels)
        self.input_channels = model_config.input_channels
        self.fuse_channels = model_config.fuse_channels
        self.intermediate_outputs = model_config.intermediate_outputs
        self.num_input_blocks = model_config.num_input_blocks
        self.shared_weights = model_config.shared_weights
        self.fuse_stride = model_config.fuse_stride

        # in case of multi input net, each input encoder will be a copy of each other
        model_config_s = model_config.clone()
        model_config_s.input_channels = model_config.input_channels[0]
        model = Model(model_config=model_config_s)

        copy_attributes = [n for n, _ in model.named_children()]
        for attr in copy_attributes:
            if hasattr(model, attr):
                val = getattr(model, attr, None)
                setattr(self, attr, val)

        if self.num_inputs>1:
            self.features = self.create_multi_input_features(self.features, self.num_inputs, self.num_input_blocks,
                                                        self.fuse_channels, self.shared_weights, self.fuse_stride)

        self._initialize_weights()

        if model_config.num_inputs>1 and pretrained:
            change_names_dict = {'^features.': ['features.stream{}.'.format(stream) for stream in range(model_config.num_inputs)]}
            xnn.utils.load_weights(self, model_config.pretrained, change_names_dict, ignore_size=True, verbose=True)
        elif pretrained:
            xnn.utils.load_weights(self, model_config.pretrained, change_names_dict=None, ignore_size=True, verbose=True)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                if m.weight is not None:
                    torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def forward(self, x):
        if self.num_inputs>1:
            x, outputs = self.forward_multi_input_features(x, self.features, self.num_inputs, self.num_input_blocks,
                                                      self.fuse_channels, self.shared_weights)
        else:
            # TODO: Cleanup. It should not be done in this complicated way.
            # To print the correct size of features.
            outputs = []
            x = x[0] if xnn.utils.is_list(x) else x
            for block_id, block in enumerate(self.features):
                if isinstance(block, torch.nn.AdaptiveAvgPool2d):
                    xnn.utils.print_once('=> feature size is: ', x.size())
                #
                x = block(x)
                outputs += [x]

        if self.num_classes is not None:
            x = torch.flatten(x, 1)
            x = self.classifier(x)

        if self.intermediate_outputs:
            return x, outputs
        else:
            return x


    def create_multi_input_features(self, features, num_inputs, num_input_blocks, fuse_channels, shared_weights, fuse_stride):
        if num_inputs == 1 or shared_weights:
            return features

        features_mi = torch.nn.ModuleDict()
        for stream_idx in range(num_inputs):
            feature_stream = []
            for block_id in range(num_input_blocks):
                block = features[block_id] if (stream_idx == 0) else copy.deepcopy(features[block_id])
                feature_stream.append(block)
            if stream_idx == 0:
                feature_stream.extend(features[num_input_blocks:])
            #
            stream_name = 'stream'+str(stream_idx)
            features_mi[stream_name] = torch.nn.Sequential(*feature_stream)

        features_mi['streamfuse'] = xnn.layers.ConvDWSepNormAct2d(fuse_channels*num_inputs, fuse_channels, stride=fuse_stride, kernel_size=3, activation=(None,None))

        return features_mi


    def forward_multi_input_features(self, x, features_mi, num_inputs, num_input_blocks, fuse_channels, shared_weights):
        outputs = []
        if num_inputs>1:
            if isinstance(x, (list, tuple)):
                assert len(x) == num_inputs, 'incorrect input. number of inputs do not match'
            else:
                assert x.size(1) == num_inputs*3, 'incorrect input. size of input does not match'
                x = xnn.layers.functional.channel_split_by_chunks(x, num_inputs)

            # shallow copy, just to create a new list
            x = list(x)

            if shared_weights:
                for stream_idx in range(num_inputs):
                    for block_index, block in enumerate(features_mi[:num_input_blocks]):
                        x[stream_idx] = block(x[stream_idx])
                        if stream_idx == 0:
                            outputs += [[x[stream_idx]]]
                        else:
                            outputs[block_index] += [x[stream_idx]]
                fuse_layer = features_mi['streamfuse']
                x = torch.cat(x, dim=1)
                x = fuse_layer(x)

                outputs += [x]

                for block in features_mi[num_input_blocks:]:
                    x = block(x)
                    outputs += [x]

            else:
                for stream_idx in range(num_inputs):
                    stream_name = 'stream' + str(stream_idx)
                    stream = features_mi[stream_name]
                    for block_index, block in enumerate(stream[:num_input_blocks]):
                        x[stream_idx] = block(x[stream_idx])
                        if stream_idx == 0:
                            outputs += [[x[stream_idx]]]
                        else:
                            outputs[block_index] += [x[stream_idx]]

                fuse_layer = features_mi['streamfuse']
                x = torch.cat(x, dim=1)
                x = fuse_layer(x)

                outputs += [x]

                stream0 = features_mi['stream0']
                for block in stream0[num_input_blocks:]:
                    x = block(x)
                    outputs += [x]
            #
        else:
            for block in features_mi:
                x = block(x)
                outputs += [x]
        #

        return x, outputs


###################################################
# these are the real multi input blocks
class MobileNetV2TVMI4(MultiInputNet):
    def __init__(self, model_config):
        model_config.num_input_blocks = 4
        model_config.fuse_channels = 24
        super().__init__(MobileNetV2TV, model_config)
#
mobilenet_v2_tv_mi4 = MobileNetV2TVMI4


# these are the real multi input blocks
class MobileNetV2EricsunMI4(MultiInputNet):
    def __init__(self, model_config):
        model_config.num_input_blocks = 4
        model_config.fuse_channels = 24
        super().__init__(MobileNetV2Ericsun, model_config)
#
mobilenet_v2_ericsun_mi4 = MobileNetV2EricsunMI4


# these are the real multi input blocks
class MobileNetV2TVNV12MI4(MultiInputNet):
    def __init__(self, model_config):
        model_config.num_input_blocks = 4
        model_config.fuse_channels = 24
        super().__init__(MobileNetV2TVNV12, model_config)
#
mobilenet_v2_tv_nv12_mi4 = MobileNetV2TVNV12MI4

# these are the real multi input blocks
class MobileNetV2TVGWSMI4(MultiInputNet):
    def __init__(self, model_config):
        model_config.num_input_blocks = 4
        model_config.fuse_channels = 24
        super().__init__(MobileNetV2TVGWS, model_config)
#
mobilenet_v2_tv_gws_mi4 = MobileNetV2TVGWSMI4


###################################################
# thes are multi input blocks, but their num_input_blocks is set to 0
class ResNet50MI4(MultiInputNet):
    def __init__(self, model_config):
        model_config.num_input_blocks = 4
        model_config.fuse_channels = 64
        super().__init__(resnet50_with_model_config, model_config)


###################################################
# thes are multi input blocks, but their num_input_blocks is set to 0
class RegNetX400MFMI4(MultiInputNet):
    def __init__(self, model_config):
        model_config.num_input_blocks = 2
        model_config.fuse_channels = 64
        super().__init__(regnetx400mf_with_model_config, model_config)


# thes are multi input blocks, but their num_input_blocks is set to 0
class RegNetX800MFMI4(MultiInputNet):
    def __init__(self, model_config):
        model_config.num_input_blocks = 2
        model_config.fuse_channels = 64
        super().__init__(regnetx800mf_with_model_config, model_config)


# thes are multi input blocks, but their num_input_blocks is set to 0
class RegNetX1p6GFMI4(MultiInputNet):
    def __init__(self, model_config):
        model_config.num_input_blocks = 2
        model_config.fuse_channels = 64
        super().__init__(regnetx1p6gf_with_model_config, model_config)


# thes are multi input blocks, but their num_input_blocks is set to 0
class RegNetX3p2GFMI4(MultiInputNet):
    def __init__(self, model_config):
        model_config.num_input_blocks = 2
        model_config.fuse_channels = 64
        super().__init__(regnetx3p2gf_with_model_config, model_config)