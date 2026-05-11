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
Reference:

Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation,
Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam,
Google Inc., https://arxiv.org/pdf/1802.02611.pdf
"""

import torch
import numpy as np

from edgeai_torchmodelopt import xnn
from .pixel2pixelnet import *

try: from .pixel2pixelnet_internal import *
except: pass

from ..multi_input_net import MobileNetV2TVMI4, MobileNetV2EricsunMI4, \
                              ResNet50MI4, RegNetX800MFMI4

###########################################
__all__ = ['DeepLabV3PlusEdgeAILite', 'DeepLabV3PlusEdgeAILiteDecoder',
           'deeplabv3plus_mobilenetv2_tv_edgeailite', 'deeplabv3plus_mobilenetv2_tv_fd_edgeailite',
           'deeplabv3plus_mobilenetv2_tv_1p4_edgeailite',
           'deeplabv3plus_mobilenetv2_ericsun_edgeailite',
           'deeplabv3plus_resnet50_edgeailite', 'deeplabv3plus_resnet50_p5_edgeailite', 'deeplabv3plus_resnet50_p5_fd_edgeailite',
           'deeplabv3plus_regnetx800mf_edgeailite', 'deeplabv3plus_regnetx800mf_bgr_edgeailite']


###########################################
class DeepLabV3PlusEdgeAILiteDecoder(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.model_config = model_config

        current_channels = model_config.shortcut_channels[-1]
        decoder_channels = round(model_config.decoder_chan*model_config.decoder_factor)
        aspp_channels = round(model_config.aspp_chan*model_config.decoder_factor)

        if model_config.use_aspp:
            group_size_dw = model_config.group_size_dw if hasattr(model_config, 'group_size_dw') else None
            ASPPBlock = xnn.layers.GWASPPLiteBlock if model_config.groupwise_sep else xnn.layers.DWASPPLiteBlock
            self.aspp = ASPPBlock(current_channels, aspp_channels, decoder_channels, dilation=model_config.aspp_dil,
                                  activation=model_config.activation, linear_dw=model_config.linear_dw,
                                  group_size_dw=group_size_dw)
        else:
            self.aspp = None

        current_channels = decoder_channels if model_config.use_aspp else current_channels

        short_chan = model_config.shortcut_channels[0]
        self.shortcut = xnn.layers.ConvNormAct2d(short_chan, model_config.shortcut_out, kernel_size=1, activation=model_config.activation)

        self.decoder_channels = merged_channels = (current_channels+model_config.shortcut_out)

        upstride1 = model_config.shortcut_strides[-1]//model_config.shortcut_strides[0]
        # use UpsampleWithGeneric() instead of UpsampleWith() to break down large upsampling factors to multiples of 4 and 2 -
        # useful if upsampling factors other than 4 and 2 are not supported.
        self.upsample1 = xnn.layers.UpsampleWith(decoder_channels, decoder_channels, upstride1, model_config.interpolation_type, model_config.interpolation_mode)

        self.cat = xnn.layers.CatBlock()

        UpsampleClass = xnn.layers.UpsampleWith if not model_config.slice_final_resize else UpsampleWithSlice

        # add prediction & upsample modules
        if self.model_config.final_prediction:
            add_lite_prediction_modules(self, model_config, merged_channels, module_names=('pred','upsample2'), UpsampleClass=UpsampleClass)
        #


    # the upsampling is using functional form to support size based upsampling for odd sizes
    # that are not a perfect ratio (eg. 257x513), which seem to be popular for segmentation
    def forward(self, x, x_features, x_list):
        assert isinstance(x, (list,tuple)) and len(x)<=2, 'incorrect input'

        x_input = x[0]
        in_shape = x_input[0].shape if isinstance(x_input, (list,tuple)) else x_input.shape

        # high res shortcut
        shape_s = xnn.utils.get_shape_with_stride(in_shape, self.model_config.shortcut_strides[0])
        shape_s[1] = self.model_config.shortcut_channels[0]
        x_s = xnn.utils.get_blob_from_list(x_list, shape_s)
        x_s = self.shortcut(x_s)

        if self.model_config.freeze_encoder:
            x_s = x_s.detach()
            x_features = x_features.detach()

        # aspp/scse blocks at output stride
        x = self.aspp(x_features) if self.model_config.use_aspp else x_features

        # upsample low res features to match with shortcut
        x = self.upsample1(x)

        # combine and do high res prediction
        x = self.cat((x,x_s))

        if self.model_config.final_prediction:
            x = self.pred(x)

            if self.model_config.final_upsample:
                x = self.upsample2(x)

            if (not self.training) and (self.model_config.output_type == 'segmentation'):
                x = torch.argmax(x, dim=1, keepdim=True)

            assert int(in_shape[2]) == int(x.shape[2]*self.model_config.target_input_ratio) and \
                   int(in_shape[3]) == int(x.shape[3]*self.model_config.target_input_ratio), 'incorrect output shape'

        if self.model_config.freeze_decoder:
            x = x.detach()

        return x


class DeepLabV3PlusEdgeAILite(Pixel2PixelNet):
    def __init__(self, base_model, model_config):
        super().__init__(base_model, DeepLabV3PlusEdgeAILiteDecoder, model_config)


###########################################
# config settings
def get_config_deeplav3lite_mnv2():
    # use list for entries that are different for different decoders.
    # and are expected to be passed from the main script.
    model_config = xnn.utils.ConfigNode()
    model_config.num_classes = None
    model_config.num_decoders = None
    model_config.input_channels = (3,)
    model_config.output_channels = [19]
    model_config.intermediate_outputs = True
    model_config.normalize_input = False
    model_config.split_outputs = False
    model_config.use_aspp = True
    model_config.fastdown = False
    model_config.target_input_ratio = 1
    model_config.prediction_channels = None

    model_config.strides = (2,2,2,2,1)
    model_config.fastdown = False
    model_config.groupwise_sep = False
    encoder_stride = np.prod(model_config.strides)
    model_config.shortcut_strides = (4,encoder_stride)
    model_config.shortcut_channels = (24,320) # this is for mobilenetv2 - change for other networks
    model_config.shortcut_out = 48
    model_config.decoder_chan = 256
    model_config.aspp_chan = 256
    model_config.aspp_dil = (6,12,18)
    model_config.final_prediction = True
    model_config.final_upsample = True
    model_config.output_range = None
    model_config.decoder_factor = 1.0
    model_config.output_type = None
    model_config.activation = xnn.layers.DefaultAct2d
    model_config.interpolation_type = 'upsample'
    model_config.interpolation_mode = 'bilinear'
    model_config.linear_dw = False
    model_config.normalize_gradients = False
    model_config.freeze_encoder = False
    model_config.freeze_decoder = False
    model_config.multi_task = False
    #Enable if want to slice final resize. Comes handy in teh case of avoiding inference for one of the slices.
    model_config.slice_final_resize = False
    return model_config


def deeplabv3plus_mobilenetv2_tv_1p4_edgeailite(model_config=None, pretrained=None):
    model_config = get_config_deeplav3lite_mnv2().merge_from(model_config)
    #for mobv2_1.4
    model_config.width_mult = 1.4
    model_config.activation = torch.nn.ReLU6

    w_m = model_config.width_mult 
    model_config.shortcut_channels = tuple([xnn.utils.make_divisible_by8(ch*w_m) for ch in model_config.shortcut_channels])
    model_config.shortcut_out = xnn.utils.make_divisible_by8(model_config.shortcut_out*w_m)
    model_config.decoder_chan = xnn.utils.make_divisible_by8(model_config.decoder_chan*w_m)
    model_config.aspp_chan = xnn.utils.make_divisible_by8(model_config.aspp_chan*w_m)
    
    return deeplabv3plus_mobilenetv2_tv_edgeailite(model_config, pretrained=pretrained)

def deeplabv3plus_mobilenetv2_tv_edgeailite(model_config=None, pretrained=None):
    model_config = get_config_deeplav3lite_mnv2().merge_from(model_config)
    # encoder setup
    model_config_e = model_config.clone()
    base_model = MobileNetV2TVMI4(model_config_e)
    # decoder setup
    model = DeepLabV3PlusEdgeAILite(base_model, model_config)

    num_inputs = len(model_config.input_channels)
    num_decoders = len(model_config.output_channels) if (
                model_config.num_decoders is None) else model_config.num_decoders
    if num_inputs > 1:
        change_names_dict = {'^features.': ['encoder.features.stream{}.'.format(stream) for stream in range(num_inputs)],
                            '^encoder.features.': ['encoder.features.stream{}.'.format(stream) for stream in range(num_inputs)],
                            '^decoders.0.': ['decoders.{}.'.format(d) for d in range(num_decoders)]}
    else:
        change_names_dict = {'^features.': 'encoder.features.',
                             '^decoders.0.': ['decoders.{}.'.format(d) for d in range(num_decoders)]}
    #

    if pretrained:
        model = xnn.utils.load_weights(model, pretrained, change_names_dict, ignore_size=True, verbose=True)

    return model, change_names_dict


def deeplabv3plus_mobilenetv2_tv_fd_edgeailite(model_config=None, pretrained=None):
    model_config = get_config_deeplav3lite_mnv2().merge_from(model_config)
    model_config.fastdown = True
    model_config.strides = (2,2,2,2,1)
    model_config.shortcut_strides = (8,32)
    model_config.shortcut_channels = (24,320)
    model_config.decoder_chan = 256
    model_config.aspp_chan = 256
    return deeplabv3plus_mobilenetv2_tv_edgeailite(model_config, pretrained=pretrained)


def deeplabv3plus_mobilenetv2_ericsun_edgeailite(model_config=None, pretrained=None):
    model_config = get_config_deeplav3lite_mnv2().merge_from(model_config)
    # encoder setup
    model_config_e = model_config.clone()
    base_model = MobileNetV2EricsunMI4(model_config_e)
    # decoder setup
    model = DeepLabV3PlusEdgeAILite(base_model, model_config)

    num_inputs = len(model_config.input_channels)
    num_decoders = len(model_config.output_channels) if (model_config.num_decoders is None) else model_config.num_decoders
    if num_inputs > 1:
        change_names_dict = {
            '^features.': ['encoder.features.stream{}.'.format(stream) for stream in range(num_inputs)],
            '^encoder.features.': ['encoder.features.stream{}.'.format(stream) for stream in range(num_inputs)],
            '^decoders.0.': ['decoders.{}.'.format(d) for d in range(num_decoders)]}
    else:
        change_names_dict = {'^features.': 'encoder.features.',
                             '^decoders.0.': ['decoders.{}.'.format(d) for d in range(num_decoders)]}
    #

    if pretrained:
        model = xnn.utils.load_weights(model, pretrained, change_names_dict, ignore_size=True, verbose=True)

    return model, change_names_dict



###########################################
# config settings for mobilenetv2 backbone
def get_config_deeplav3lite_resnet50():
    # only the delta compared to the one defined for mobilenetv2
    model_config = get_config_deeplav3lite_mnv2()
    model_config.shortcut_channels = (256,2048)
    return model_config


def deeplabv3plus_resnet50_edgeailite(model_config=None, pretrained=None):
    model_config = get_config_deeplav3lite_resnet50().merge_from(model_config)
    # encoder setup
    model_config_e = model_config.clone()
    base_model = ResNet50MI4(model_config_e)
    # decoder setup
    model = DeepLabV3PlusEdgeAILite(base_model, model_config)

    # the pretrained model provided by torchvision and what is defined here differs slightly
    # note: that this change_names_dict  will take effect only if the direct load fails
    # finally take care of the change for deeplabv3plus_edgeailite (features->encoder.features)
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


def deeplabv3plus_resnet50_p5_edgeailite(model_config=None, pretrained=None):
    model_config.width_mult = 0.5
    model_config.shortcut_channels = (128,1024)
    return deeplabv3plus_resnet50_edgeailite(model_config, pretrained=pretrained)


def deeplabv3plus_resnet50_p5_fd_edgeailite(model_config, pretrained=None):
    model_config.width_mult = 0.5
    model_config.fastdown = True
    model_config.shortcut_channels = (128,1024)
    model_config.shortcut_strides = (8,64)
    return deeplabv3plus_resnet50_edgeailite(model_config, pretrained=pretrained)


###########################################
# config settings for mobilenetv2 backbone
def get_config_deeplav3lite_regnetx800mf():
    # only the delta compared to the one defined for mobilenetv2
    model_config = get_config_deeplav3lite_mnv2()
    model_config.shortcut_channels = (64,672)
    model_config.group_size_dw = 16
    return model_config


# here this is nothing specific about bgr in this model
# but is just a reminder that regnet models are typically trained with bgr input
def deeplabv3plus_regnetx800mf_edgeailite(model_config=None, pretrained=None):
    model_config = get_config_deeplav3lite_regnetx800mf().merge_from(model_config)
    # encoder setup
    model_config_e = model_config.clone()
    base_model = RegNetX800MFMI4(model_config_e)
    # decoder setup
    model = DeepLabV3PlusEdgeAILite(base_model, model_config)

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
            xnn.utils.load_weights(model, pretrained, change_names_dict=change_names_dict, ignore_size=ignore_size, verbose=verbose,
                                           state_dict_name=state_dict_name)
        #
        model.load_weights = load_weights_func

    return model, change_names_dict

deeplabv3plus_regnetx800mf_bgr_edgeailite = deeplabv3plus_regnetx800mf_edgeailite