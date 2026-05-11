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

"""
Reference: https://arxiv.org/abs/1505.04597

U-Net: Convolutional Networks for Biomedical Image Segmentation
Olaf Ronneberger, Philipp Fischer, Thomas Brox
"""

import torch
import numpy as np
from edgeai_torchmodelopt import xnn

from .pixel2pixelnet import *
from ..multi_input_net import MobileNetV2TVMI4, ResNet50MI4, RegNetX800MFMI4


__all__ = ['UNetASPPEdgeAILite', 'UNetEdgeAILiteDecoder',
           'unet_aspp_mobilenetv2_tv_edgeailite', 'unet_aspp_mobilenetv2_tv_fd_edgeailite',
           'unet_aspp_resnet50_edgeailite', 'unet_aspp_resnet50_fd_edgeailite',
           'unet_aspp_regnetx800mf_edgeailite', 'unet_aspp_regnetx800mf_bgr_edgeailite'
           ]

# config settings for mobilenetv2 backbone
def get_config_unet_mnv2_edgeailite():
    model_config = xnn.utils.ConfigNode()
    model_config.num_classes = None
    model_config.num_decoders = None
    model_config.intermediate_outputs = True
    model_config.use_aspp = True
    model_config.use_extra_strides = False
    model_config.groupwise_sep = False
    model_config.fastdown = False
    model_config.width_mult = 1.0
    model_config.target_input_ratio = 1
    model_config.input_channels = (3,)
    model_config.prediction_channels = None

    model_config.strides = (2,2,2,2,2)
    encoder_stride = np.prod(model_config.strides)
    model_config.shortcut_strides = (2,4,8,16,encoder_stride)
    model_config.shortcut_channels = (16,24,32,96,320) # this is for mobilenetv2 - change for other networks
    model_config.decoder_chan = 256
    model_config.aspp_chan = 256
    model_config.aspp_dil = (6,12,18)

    model_config.kernel_size_smooth = 3
    model_config.interpolation_type = 'upsample'
    model_config.interpolation_mode = 'bilinear'

    model_config.final_prediction = True
    model_config.final_upsample = True
    model_config.output_range = None

    model_config.normalize_input = False
    model_config.split_outputs = False
    model_config.decoder_factor = 1.0
    model_config.activation = xnn.layers.DefaultAct2d
    model_config.linear_dw = False
    model_config.normalize_gradients = False
    model_config.freeze_encoder = False
    model_config.freeze_decoder = False
    model_config.multi_task = False
    return model_config


###########################################
class UNetEdgeAILitePyramid(torch.nn.Module):
    def __init__(self, current_channels, minimum_channels, shortcut_strides, shortcut_channels, activation,
                 kernel_size_smooth, interpolation_type, interpolation_mode, group_size_dw=None):
        super().__init__()
        self.shortcut_strides = shortcut_strides
        self.shortcut_channels = shortcut_channels
        self.upsamples = torch.nn.ModuleList()
        self.concats = torch.nn.ModuleList()
        self.smooth_convs = torch.nn.ModuleList()

        self.smooth_convs.append(None)
        self.concats.append(None)

        upstride = 2
        activation2 = (activation, activation)
        for idx, (s_stride, feat_chan) in enumerate(zip(shortcut_strides, shortcut_channels)):
            self.upsamples.append(xnn.layers.UpsampleWith(current_channels, current_channels, upstride,
                                                          interpolation_type, interpolation_mode))
            self.concats.append(xnn.layers.CatBlock())
            smooth_channels = max(minimum_channels, feat_chan)
            self.smooth_convs.append(xnn.layers.ConvDWSepNormAct2d(current_channels+feat_chan, smooth_channels,
                                kernel_size=kernel_size_smooth, activation=activation2, group_size_dw=group_size_dw))
            current_channels = smooth_channels
        #
    #


    def forward(self, x_input, x_list):
        in_shape = x_input.shape
        x = x_list[-1]

        outputs = []

        x = self.smooth_convs[0](x) if (self.smooth_convs[0] is not None) else x
        outputs.append(x)

        for idx, (concat, smooth_conv, s_stride, short_chan, upsample) in \
                enumerate(zip(self.concats[1:], self.smooth_convs[1:], self.shortcut_strides, self.shortcut_channels, self.upsamples)):
            # get the feature of lower stride
            shape_s = xnn.utils.get_shape_with_stride(in_shape, s_stride)
            shape_s[1] = short_chan
            x_s = xnn.utils.get_blob_from_list(x_list, shape_s)
            # upsample current output and concat to that
            x = upsample(x)
            x = concat((x,x_s)) if (concat is not None) else x
            # smooth conv
            x = smooth_conv(x) if (smooth_conv is not None) else x
            # output
            outputs.append(x)
        #
        return outputs[::-1]


###########################################
class UNetEdgeAILiteDecoder(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        activation = self.model_config.activation
        self.output_type = model_config.output_type
        self.decoder_channels = decoder_channels = round(self.model_config.decoder_chan*self.model_config.decoder_factor)
        group_size_dw = model_config.group_size_dw if hasattr(model_config, 'group_size_dw') else None

        self.rfblock = None
        if self.model_config.use_aspp:
            current_channels = self.model_config.shortcut_channels[-1]
            aspp_channels = round(self.model_config.aspp_chan * self.model_config.decoder_factor)
            self.rfblock = xnn.layers.DWASPPLiteBlock(current_channels, aspp_channels, decoder_channels,
                                    dilation=self.model_config.aspp_dil, avg_pool=False, activation=activation,
                                    group_size_dw=group_size_dw)
            current_channels = decoder_channels
        elif self.model_config.use_extra_strides:
            # a low complexity pyramid
            current_channels = self.model_config.shortcut_channels[-3]
            self.rfblock = torch.nn.Sequential(
                xnn.layers.ConvDWSepNormAct2d(current_channels, current_channels, kernel_size=3, stride=2,
                                              activation=(activation, activation), group_size_dw=group_size_dw),
                xnn.layers.ConvDWSepNormAct2d(current_channels, decoder_channels, kernel_size=3, stride=2,
                                              activation=(activation, activation), group_size_dw=group_size_dw))
            current_channels = decoder_channels
        else:
            current_channels = self.model_config.shortcut_channels[-1]
            self.rfblock = xnn.layers.ConvNormAct2d(current_channels, decoder_channels, kernel_size=1, stride=1)
            current_channels = decoder_channels
        #

        minimum_channels = max(self.model_config.output_channels*2, 32)
        shortcut_strides = self.model_config.shortcut_strides[::-1][1:]
        shortcut_channels = self.model_config.shortcut_channels[::-1][1:]
        self.unet = UNetEdgeAILitePyramid(current_channels, minimum_channels, shortcut_strides, shortcut_channels,
                           self.model_config.activation, self.model_config.kernel_size_smooth,
                           self.model_config.interpolation_type, self.model_config.interpolation_mode,
                           group_size_dw=group_size_dw)
        current_channels = max(minimum_channels, shortcut_channels[-1])

        # add prediction & upsample modules
        if self.model_config.final_prediction:
            add_lite_prediction_modules(self, model_config, current_channels, module_names=('pred','upsample'))
        #


    def forward(self, x_input, x, x_list):
        assert isinstance(x_input, (list,tuple)) and len(x_input)<=2, 'incorrect input'
        assert x is x_list[-1], 'the features must the last one in x_list'
        x_input = x_input[0]
        in_shape = x_input.shape

        if self.model_config.use_extra_strides:
            for blk in self.rfblock:
                x = blk(x)
                x_list += [x]
            #
        elif self.rfblock is not None:
            x = self.rfblock(x)
            x_list[-1] = x
        #

        x_list = self.unet(x_input, x_list)
        x = x_list[0]

        if self.model_config.final_prediction:
            # prediction
            x = self.pred(x)

            # final prediction is the upsampled one
            if self.model_config.final_upsample:
                x = self.upsample(x)

            if (not self.training) and (self.output_type == 'segmentation'):
                x = torch.argmax(x, dim=1, keepdim=True)

            assert int(in_shape[2]) == int(x.shape[2]*self.model_config.target_input_ratio) and \
                   int(in_shape[3]) == int(x.shape[3]*self.model_config.target_input_ratio), 'incorrect output shape'

        return x


###########################################
class UNetASPPEdgeAILite(Pixel2PixelNet):
    def __init__(self, base_model, model_config):
        super().__init__(base_model, UNetEdgeAILiteDecoder, model_config)


###########################################
def unet_aspp_mobilenetv2_tv_edgeailite(model_config=None, pretrained=None):
    model_config = get_config_unet_mnv2_edgeailite().merge_from(model_config)
    # encoder setup
    model_config_e = model_config.clone()
    base_model = MobileNetV2TVMI4(model_config_e)
    # decoder setup
    model = UNetASPPEdgeAILite(base_model, model_config)

    num_inputs = len(model_config.input_channels)
    num_decoders = len(model_config.output_channels) if (model_config.num_decoders is None) else model_config.num_decoders
    if num_inputs > 1:
        change_names_dict = {'^features.': ['encoder.features.stream{}.'.format(stream) for stream in range(num_inputs)],
                            '^classifier.': 'encoder.classifier.',
                            '^encoder.features.': ['encoder.features.stream{}.'.format(stream) for stream in range(num_inputs)],
                            '^decoders.0.': ['decoders.{}.'.format(d) for d in range(num_decoders)]}
    else:
        change_names_dict = {'^features.': 'encoder.features.',
                             '^classifier.': 'encoder.classifier.',
                             '^decoders.0.': ['decoders.{}.'.format(d) for d in range(num_decoders)]}
    #

    if pretrained:
        model = xnn.utils.load_weights(model, pretrained, change_names_dict, ignore_size=True, verbose=True)
    #
    return model, change_names_dict


# fast down sampling model (encoder stride 64 model)
def unet_aspp_mobilenetv2_tv_fd_edgeailite(model_config=None, pretrained=None):
    model_config = get_config_unet_mnv2_edgeailite().merge_from(model_config)
    model_config.fastdown = True
    model_config.strides = (2,2,2,2,2)
    model_config.shortcut_strides = (4,8,16,32,64)
    model_config.shortcut_channels = (16,24,32,96,320)
    model_config.decoder_chan = 256
    model_config.aspp_chan = 256
    return unet_aspp_mobilenetv2_tv_edgeailite(model_config, pretrained=pretrained)


###########################################
def get_config_unet_resnet50_edgeailite():
    # only the delta compared to the one defined for mobilenetv2
    model_config = get_config_unet_mnv2_edgeailite()
    model_config.shortcut_strides = (2,4,8,16,32)
    model_config.shortcut_channels = (64,256,512,1024,2048)
    return model_config


def unet_aspp_resnet50_edgeailite(model_config, pretrained=None):
    model_config = get_config_unet_resnet50_edgeailite().merge_from(model_config)
    # encoder setup
    model_config_e = model_config.clone()
    base_model = ResNet50MI4(model_config_e)
    # decoder setup
    model = UNetASPPEdgeAILite(base_model, model_config)

    # the pretrained model provided by torchvision and what is defined here differs slightly
    # note: that this change_names_dict  will take effect only if the direct load fails
    # finally take care of the change for unet (features->encoder.features)
    num_inputs = len(model_config.input_channels)
    num_decoders = len(model_config.output_channels) if (model_config.num_decoders is None) else model_config.num_decoders
    if num_inputs > 1:
        change_names_dict = {'^conv1.': ['encoder.features.stream{}.conv1.'.format(stream) for stream in range(num_inputs)],
                            '^bn1.': ['encoder.features.stream{}.bn1.'.format(stream) for stream in range(num_inputs)],
                            '^relu.': ['encoder.features.stream{}.relu.'.format(stream) for stream in range(num_inputs)],
                            '^maxpool.': ['encoder.features.stream{}.maxpool.'.format(stream) for stream in range(num_inputs)],
                            '^layer': ['encoder.features.stream{}.layer'.format(stream) for stream in range(num_inputs)],
                            '^features.': ['encoder.features.stream{}.'.format(stream) for stream in range(num_inputs)],
                            '^encoder.features.': ['encoder.features.stream{}.'.format(stream) for stream in range(num_inputs)],
                            '^decoders.0.': ['decoders.{}.'.format(d) for d in range(num_decoders)]}
    else:
        change_names_dict = {'^conv1.': 'encoder.features.conv1.',
                             '^bn1.': 'encoder.features.bn1.',
                             '^relu.': 'encoder.features.relu.',
                             '^maxpool.': 'encoder.features.maxpool.',
                             '^layer': 'encoder.features.layer',
                             '^features.': 'encoder.features.',
                             '^decoders.0.': ['decoders.{}.'.format(d) for d in range(num_decoders)]}
    #

    if pretrained:
        model = xnn.utils.load_weights(model, pretrained, change_names_dict, ignore_size=True, verbose=True)

    return model, change_names_dict


def unet_aspp_resnet50_fd_edgeailite(model_config=None, pretrained=None):
    model_config = get_config_unet_resnet50_edgeailite().merge_from(model_config)
    model_config.fastdown = True
    model_config.strides = (2,2,2,2,2)
    model_config.shortcut_strides = (2,4,8,16,32,64) #(4,8,16,32,64)
    model_config.shortcut_channels = (64,64,256,512,1024,2048) #(64,256,512,1024,2048)
    model_config.decoder_chan = 256 #128
    model_config.aspp_chan = 256 #128
    return unet_aspp_resnet50_edgeailite(model_config, pretrained=pretrained)


###########################################
# config settings for mobilenetv2 backbone
def get_config_unet_regnetx800mf_edgeailite():
    # only the delta compared to the one defined for mobilenetv2
    model_config = get_config_unet_mnv2_edgeailite()
    model_config.shortcut_strides = (2,4,8,16,32)
    model_config.shortcut_channels = (32,64,128,288,672)
    model_config.group_size_dw = 16
    return model_config


# here this is nothing specific about bgr in this model
# but is just a reminder that regnet models are typically trained with bgr input
def unet_aspp_regnetx800mf_edgeailite(model_config=None, pretrained=None):
    model_config = get_config_unet_regnetx800mf_edgeailite().merge_from(model_config)
    # encoder setup
    model_config_e = model_config.clone()
    base_model = RegNetX800MFMI4(model_config_e)
    # decoder setup
    model = UNetASPPEdgeAILite(base_model, model_config)

    # the pretrained model provided by torchvision and what is defined here differs slightly
    # note: that this change_names_dict  will take effect only if the direct load fails
    # finally take care of the change for deeplabv3plus_edgeailite (features->encoder.features)
    num_inputs = len(model_config.input_channels)
    num_decoders = len(model_config.output_channels) if (model_config.num_decoders is None) else model_config.num_decoders
    if num_inputs > 1:
        change_names_dict = {'^features.': ['encoder.features.stream{}.'.format(stream) for stream in range(num_inputs)],
                             '^encoder.features.': ['encoder.features.stream{}.'.format(stream) for stream in range(num_inputs)],
                             '^decoders.0.': ['decoders.{}.'.format(d) for d in range(num_decoders)]}
    else:
        change_names_dict = {'^stem.': 'encoder.features.stem.',
                             '^s1': 'encoder.features.s1', '^s2': 'encoder.features.s2',
                             '^s3': 'encoder.features.s3', '^s4': 'encoder.features.s4',
                             '^features.': 'encoder.features.',
                             '^decoders.0.': ['decoders.{}.'.format(d) for d in range(num_decoders)]}
    #

    if pretrained:
        model = xnn.utils.load_weights(model, pretrained, change_names_dict, ignore_size=True, verbose=True,
                                       state_dict_name=['state_dict','model_state'])
    else:
        # need to use state_dict_name as the checkpoint uses a different name for state_dict
        # provide a custom load_weighs for the model
        def load_weights_func(pretrained, change_names_dict, ignore_size=True, verbose=True,
                                       state_dict_name=['state_dict','model_state']):
            xnn.utils.load_weights(model, pretrained, change_names_dict=change_names_dict, ignore_size=ignore_size,
                                       verbose=verbose, state_dict_name=state_dict_name)
        #
        model.load_weights = load_weights_func
    #
    return model, change_names_dict

unet_aspp_regnetx800mf_bgr_edgeailite = unet_aspp_regnetx800mf_edgeailite