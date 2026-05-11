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


# Our implementation of BiFPN-Lite (i.e. without the weighting before adding tensors):
# Reference:
# EfficientDet: Scalable and Efficient Object Detection
# Mingxing Tan, Ruoming Pang, Quoc V. Le,
# Google Research, Brain Team
# https://arxiv.org/pdf/1911.09070.pdf

import copy
import torch
import numpy as np
from edgeai_torchmodelopt import xnn

from .pixel2pixelnet import *
from ..multi_input_net import MobileNetV2TVMI4, ResNet50MI4, \
    RegNetX400MFMI4, RegNetX800MFMI4, RegNetX1p6GFMI4, RegNetX3p2GFMI4


__all__ = ['BiFPNASPPEdgeAILite', 'BiFPNEdgeAILiteDecoder',
           'bifpn_aspp_mobilenetv2_tv_edgeailite', 'bifpn_mobilenetv2_tv_edgeailite',
           'bifpn_aspp_regnetx400mf_edgeailite', 'bifpn_aspp_regnetx400mf_bgr_edgeailite',
           'bifpn_aspp_regnetx800mf_edgeailite', 'bifpn_aspp_regnetx800mf_bgr_edgeailite',
           'bifpn_aspp_regnetx1p6gf_edgeailite', 'bifpn_aspp_regnetx1p6gf_bgr_edgeailite',
           'bifpn_aspp_regnetx3p2gf_edgeailite', 'bifpn_aspp_regnetx3p2gf_bgr_edgeailite'
           ]

# config settings for mobilenetv2 backbone
def get_config_bifpn_mnv2_edgeailite():
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

    model_config.num_bifpn_blocks = 4
    model_config.num_head_blocks = 0 #1
    model_config.num_fpn_outs = 6
    model_config.strides = (2,2,2,2,2)
    encoder_stride = np.prod(model_config.strides)
    model_config.shortcut_strides = (4,8,16,encoder_stride)
    # this is for mobilenetv2 - change for other networks
    model_config.shortcut_channels = (24,32,96,320)

    model_config.aspp_dil = (6,12,18)
    model_config.decoder_chan = 128 #256
    model_config.aspp_chan = 128    #256
    model_config.fpn_chan = 128     #256
    model_config.head_chan = 128    #256

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
class BiFPNEdgeAILite(torch.nn.Module):
    def __init__(self, model_config, in_channels=None, intermediate_channels=None, out_channels=None,
                 num_outs=5, num_blocks=None, add_extra_convs = 'on_output',
                 group_size_dw=None, normalization=None, activation=None, **kwargs):
        super().__init__()
        self.model_config = model_config
        self.add_extra_convs = add_extra_convs
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.num_fpn_outs = num_outs # for now set them to be same - but can be different

        blocks = []
        for i in range(num_blocks):
            last_in_channels = [intermediate_channels for _ in range(self.num_fpn_outs)] if i>0 else in_channels
            if i < (num_blocks-1):
                # the initial bifpn blocks can operate with fewer number of channels
                block_id = i
                bi_fpn = BiFPNEdgeAILiteBlock(block_id=block_id, in_channels=in_channels, out_channels=intermediate_channels, num_outs=self.num_fpn_outs,
                                group_size_dw=group_size_dw, normalization=normalization, activation=activation, **kwargs)
            else:
                # block_id=0 will cause shortcut connections to be created to change the number of channels
                block_id = 0 if ((num_blocks == 1) or (out_channels != intermediate_channels)) else i
                # for segmentation, last one doesn't need down blocks as they are not used.
                bi_fpn = BiFPNEdgeAILiteBlock(block_id=block_id, up_only=True, in_channels=last_in_channels, out_channels=out_channels,
                                        num_outs=self.num_fpn_outs, group_size_dw=group_size_dw, normalization=normalization,
                                        activation=activation, **kwargs)
            #
            blocks.append(bi_fpn)
        #
        self.bifpn_blocks = torch.nn.Sequential(*blocks)

        NormType1 = normalization[-1] if isinstance(normalization,(list,tuple)) else normalization
        self.extra_convs = torch.nn.ModuleList()
        if self.num_outs > self.num_fpn_outs:
            in_ch = self.in_channels[-1] if self.add_extra_convs == 'on_input' else self.out_channels
            DownsampleType = torch.nn.MaxPool2d
            for i in range(self.num_outs-self.num_fpn_outs):
                extra_conv = BiFPNEdgeAILiteBlock.build_downsample_module(in_ch, self.out_channels, kernel_size=3, stride=2,
                                    group_size_dw=group_size_dw, normalization=NormType1, activation=None,
                                    DownsampleType=DownsampleType)
                self.extra_convs.append(extra_conv)
                in_ch = self.out_channels
            #
        #

    def forward(self, x_input, x_list):
        in_shape = x_input.shape
        inputs = []
        for s_chan, s_stride in zip(self.in_channels, self.model_config.shortcut_strides):
            shape_s = xnn.utils.get_shape_with_stride(in_shape, s_stride)
            shape_s[1] = s_chan
            x_s = xnn.utils.get_blob_from_list(x_list, shape_s)
            inputs.append(x_s)
        #

        assert len(inputs) == len(self.in_channels)
        outputs = self.bifpn_blocks(inputs)
        outputs = list(outputs)
        if self.num_outs > self.num_fpn_outs:
            inp = inputs[-1] if self.add_extra_convs == 'on_input' else outputs[-1]
            for i in range(self.num_outs-self.num_fpn_outs):
                extra_inp = self.extra_convs[i](inp)
                outputs.append(extra_inp)
                inp = extra_inp
            #
        #
        return outputs


class BiFPNEdgeAILiteBlock(torch.nn.Module):
    def __init__(self, block_id=None, in_channels=None, out_channels=None, num_outs=None, up_only=False, start_level=0, end_level=-1,
                 add_extra_convs=None, group_size_dw=None, normalization=xnn.layers.DefaultNorm2d, activation=xnn.layers.DefaultAct2d):
        super(BiFPNEdgeAILiteBlock, self).__init__()
        assert isinstance(in_channels, (list,tuple))
        self.up_only = up_only
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.block_id = block_id
        assert block_id is not None, f'block_id must be valid: {block_id}'

        NormType = normalization
        ActType = activation
        DownsampleType = torch.nn.MaxPool2d
        UpsampleType = xnn.layers.ResizeWith
        ConvModuleWrapper = xnn.layers.ConvDWSepNormAct2d
        NormType1 = normalization[-1] if isinstance(normalization,(list,tuple)) else normalization
        ActType1 = activation[-1] if isinstance(activation,(list,tuple)) else activation
        upsample_cfg = dict(scale_factor=2, mode='bilinear')

        # add extra conv layers (e.g., RetinaNet)
        if block_id == 0:
            self.num_backbone_convs = (self.backbone_end_level - self.start_level)
            self.extra_levels = num_outs - self.num_backbone_convs
            self.in_convs = torch.nn.ModuleList()
            for i in range(num_outs):
                if i < self.num_backbone_convs:
                    in_ch = in_channels[self.start_level + i]
                elif i == self.num_backbone_convs:
                    in_ch = in_channels[-1]
                else:
                    in_ch = out_channels
                #
                stride = 1 if i < self.num_backbone_convs else 2
                in_conv = BiFPNEdgeAILiteBlock.build_downsample_module(in_ch, out_channels, kernel_size=3, stride=stride,
                                   group_size_dw=group_size_dw, normalization=NormType1, activation=None,
                                   DownsampleType=DownsampleType)
                self.in_convs.append(in_conv)
            #
        #

        self.ups = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()
        self.up_acts = torch.nn.ModuleList()
        self.up_adds = torch.nn.ModuleList()
        if not up_only:
            self.downs = torch.nn.ModuleList()
            self.down_convs = torch.nn.ModuleList()
            self.down_acts = torch.nn.ModuleList()
            self.down_adds1 = torch.nn.ModuleList()
            self.down_adds2 = torch.nn.ModuleList()
        #
        for i in range(self.num_outs-1):
            # up modules
            up = UpsampleType(**upsample_cfg)
            up_conv = ConvModuleWrapper(out_channels, out_channels, 3, padding=1,
                    group_size_dw=group_size_dw, normalization=NormType, activation=ActType)
            up_act = ActType1()
            self.ups.append(up)
            self.up_convs.append(up_conv)
            self.up_acts.append(up_act)
            self.up_adds.append(xnn.layers.AddBlock())
            # down modules
            if not up_only:
                down = DownsampleType(kernel_size=3, stride=2, padding=1)
                down_conv = ConvModuleWrapper(out_channels, out_channels, 3, padding=1,
                    group_size_dw=group_size_dw, normalization=NormType, activation=ActType)
                down_act = ActType1()
                self.downs.append(down)
                self.down_convs.append(down_conv)
                self.down_acts.append(down_act)
                self.down_adds1.append(xnn.layers.AddBlock())
                self.down_adds2.append(xnn.layers.AddBlock())
            #

    def forward(self, inputs):
        # in convs
        if self.block_id == 0:
            ins = [self.in_convs[i](inputs[self.start_level+i]) for i in range(self.num_backbone_convs)]
            extra_in = inputs[-1]
            for i in range(self.num_backbone_convs, self.num_outs):
                extra_in = self.in_convs[i](extra_in)
                ins.append(extra_in)
            #
        else:
            ins = inputs
        #
        # up convs
        ups = [None] * self.num_outs
        ups[-1] = ins[-1]
        for i in range(self.num_outs-2, -1, -1):
            add_block = self.up_adds[i]
            ups[i] = self.up_convs[i](self.up_acts[i](
                    add_block((ins[i], self.ups[i](ups[i+1])))
            ))
        #
        if self.up_only:
            return tuple(ups)
        else:
            # down convs
            outs = [None] * self.num_outs
            outs[0] = ups[0]
            for i in range(0, self.num_outs-1):
                add_block1 = self.down_adds1[i]
                res = add_block1((ins[i+1], ups[i+1])) if (ins[i+1] is not ups[i+1]) else ins[i+1]
                add_block2 = self.down_adds2[i]
                outs[i+1] = self.down_convs[i](self.down_acts[i](
                    add_block2((res,self.downs[i](outs[i])))
                ))
            #
            return tuple(outs)

    @staticmethod
    def build_downsample_module(in_channels, out_channels, kernel_size, stride,
                                group_size_dw=None, normalization=None, activation=None,
                                DownsampleType=None):
        NormType = normalization
        ActType = activation
        ConvModuleWrapper = xnn.layers.ConvNormAct2d
        padding = kernel_size//2
        if in_channels == out_channels and stride == 1:
            block = ConvModuleWrapper(in_channels, out_channels, kernel_size=1, stride=1,
                                padding=0, normalization=NormType, activation=ActType)
        elif in_channels == out_channels and stride > 1:
            block = DownsampleType(kernel_size=kernel_size, stride=stride, padding=padding)
        elif in_channels != out_channels and stride == 1:
            block = ConvModuleWrapper(in_channels, out_channels, kernel_size=1, stride=stride,
                                padding=0, normalization=NormType, activation=ActType)
        else:
            block = torch.nn.Sequential(
                    DownsampleType(kernel_size=kernel_size, stride=stride, padding=padding),
                    ConvModuleWrapper(in_channels, out_channels, kernel_size=1, stride=1,
                                padding=0, normalization=NormType, activation=ActType))
        #
        return block


###########################################
class BiFPNEdgeAILiteDecoder(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        normalization = xnn.layers.DefaultNorm2d
        normalization_dws = (normalization, normalization)
        activation = xnn.layers.DefaultAct2d
        activation_dws = (activation, activation)
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
                xnn.layers.ConvDWSepNormAct2d(current_channels, current_channels, kernel_size=3,
                        stride=2, activation=(activation, activation), group_size_dw=group_size_dw),
                xnn.layers.ConvDWSepNormAct2d(current_channels, decoder_channels, kernel_size=3,
                        stride=2, activation=(activation, activation), group_size_dw=group_size_dw))
            current_channels = decoder_channels
        else:
            current_channels = self.model_config.shortcut_channels[-1]
            self.rfblock = xnn.layers.ConvNormAct2d(current_channels, decoder_channels, kernel_size=1, stride=1)
            current_channels = decoder_channels
        #

        shortcut_strides = self.model_config.shortcut_strides
        fpn_in_channels = list(copy.deepcopy(self.model_config.shortcut_channels))
        fpn_in_channels[-1] = current_channels
        fpn_channels = round(self.model_config.fpn_chan*self.model_config.decoder_factor)
        out_channels = round(self.model_config.head_chan*self.model_config.decoder_factor)
        num_fpn_outs = self.model_config.num_fpn_outs
        num_bifpn_blocks = self.model_config.num_bifpn_blocks
        group_size_dw = self.model_config.group_size_dw if hasattr(model_config, 'group_size_dw') else None
        self.fpn = BiFPNEdgeAILite(self.model_config, in_channels=fpn_in_channels, intermediate_channels=fpn_channels,
                        out_channels=out_channels, num_outs=num_fpn_outs, num_blocks=num_bifpn_blocks,
                        group_size_dw=group_size_dw, normalization=normalization_dws, activation=activation_dws)

        head = []
        current_channels = out_channels
        if self.model_config.num_head_blocks > 0:
            for h_idx in range(self.model_config.num_head_blocks):
                hblock = xnn.layers.ConvDWSepNormAct2d(current_channels, out_channels, kernel_size=3, stride=1,
                                                       group_size_dw=group_size_dw, normalization=normalization_dws,
                                                       activation=activation_dws)
                current_channels = out_channels
                head.append(hblock)
            #
            self.head = torch.nn.Sequential(*head)
        else:
            self.head = None
        #

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

        x_list = self.fpn(x_input, x_list)
        x = x_list[0]

        x = self.head(x) if (self.head is not None) else x

        if self.model_config.final_prediction:
            # prediction
            x = self.pred(x)

            # final prediction is the upsampled one
            if self.model_config.final_upsample:
                x = self.upsample(x)

            if (not self.training) and (self.output_type == 'segmentation'):
                x = torch.argmax(x, dim=1, keepdim=True)

            assert int(in_shape[2]) == int(x.shape[2]) and int(in_shape[3]) == int(x.shape[3]), 'incorrect output shape'
        #
        return x


###########################################
class BiFPNASPPEdgeAILite(Pixel2PixelNet):
    def __init__(self, base_model, model_config):
        super().__init__(base_model, BiFPNEdgeAILiteDecoder, model_config)


###########################################
def bifpn_aspp_mobilenetv2_tv_edgeailite(model_config, pretrained=None):
    model_config = get_config_bifpn_mnv2_edgeailite().merge_from(model_config)
    # encoder setup
    model_config_e = model_config.clone()
    base_model = MobileNetV2TVMI4(model_config_e)
    # decoder setup
    model = BiFPNASPPEdgeAILite(base_model, model_config)

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



##################
# similar to the original fpn model with extra convolutions with strides (no aspp)
def bifpn_mobilenetv2_tv_edgeailite(model_config, pretrained=None):
    model_config = get_config_bifpn_mnv2_edgeailite().merge_from(model_config)
    model_config.use_aspp = False
    return bifpn_aspp_mobilenetv2_tv_edgeailite(model_config, pretrained=pretrained)


###########################################
# here this is nothing specific about bgr in this model
# but is just a reminder that regnet models are typically trained with bgr input
def bifpn_aspp_regnetx_edgeailite(model_config=None, pretrained=None, base_model_class=None):
    # encoder setup
    model_config_e = model_config.clone()
    base_model = base_model_class(model_config_e)
    # decoder setup
    model = BiFPNASPPEdgeAILite(base_model, model_config)

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


###########################################
# config settings for mobilenetv2 backbone
def get_config_bifpn_regnetx400mf_edgeailite():
    # only the delta compared to the one defined for mobilenetv2
    model_config = get_config_bifpn_mnv2_edgeailite()
    model_config.group_size_dw = 16
    model_config.shortcut_channels = (32,64,160,384)
    return model_config

def bifpn_aspp_regnetx400mf_edgeailite(model_config=None, pretrained=None):
    model_config = get_config_bifpn_regnetx400mf_edgeailite().merge_from(model_config)
    return bifpn_aspp_regnetx_edgeailite(model_config, pretrained, base_model_class=RegNetX400MFMI4)


bifpn_aspp_regnetx400mf_bgr_edgeailite = bifpn_aspp_regnetx400mf_edgeailite


###########################################
# config settings for mobilenetv2 backbone
def get_config_bifpn_regnetx800mf_edgeailite():
    # only the delta compared to the one defined for mobilenetv2
    model_config = get_config_bifpn_mnv2_edgeailite()
    model_config.group_size_dw = 16
    model_config.shortcut_channels = (64,128,288,672)
    return model_config

def bifpn_aspp_regnetx800mf_edgeailite(model_config=None, pretrained=None):
    model_config = get_config_bifpn_regnetx800mf_edgeailite().merge_from(model_config)
    return bifpn_aspp_regnetx_edgeailite(model_config, pretrained, base_model_class=RegNetX800MFMI4)


bifpn_aspp_regnetx800mf_bgr_edgeailite = bifpn_aspp_regnetx800mf_edgeailite


###########################################
# config settings for mobilenetv2 backbone
def get_config_bifpn_regnetx1p6gf_edgeailite():
    # only the delta compared to the one defined for mobilenetv2
    model_config = get_config_bifpn_mnv2_edgeailite()
    # group size is 24. make the decoder channels multiples of 24
    model_config.group_size_dw = 24
    model_config.decoder_chan = 168 #264
    model_config.aspp_chan = 168    #264
    model_config.fpn_chan = 168     #264
    model_config.head_chan = 168    #264
    model_config.shortcut_channels = (72, 168, 408, 912)
    return model_config


def bifpn_aspp_regnetx1p6gf_edgeailite(model_config=None, pretrained=None):
    model_config = get_config_bifpn_regnetx1p6gf_edgeailite().merge_from(model_config)
    return bifpn_aspp_regnetx_edgeailite(model_config, pretrained, base_model_class=RegNetX1p6GFMI4)


bifpn_aspp_regnetx1p6gf_bgr_edgeailite = bifpn_aspp_regnetx1p6gf_edgeailite


###########################################
# config settings for mobilenetv2 backbone
def get_config_bifpn_regnetx3p2gf_edgeailite():
    # only the delta compared to the one defined for mobilenetv2
    model_config = get_config_bifpn_mnv2_edgeailite()
    # group size is 48. make the decoder channels multiples of 48
    model_config.group_size_dw = 48
    model_config.decoder_chan = 192 #288
    model_config.aspp_chan = 192    #288
    model_config.fpn_chan = 192     #288
    model_config.head_chan = 192    #288
    model_config.shortcut_channels = (96, 192, 432, 1008)
    return model_config


def bifpn_aspp_regnetx3p2gf_edgeailite(model_config=None, pretrained=None):
    model_config = get_config_bifpn_regnetx3p2gf_edgeailite().merge_from(model_config)
    return bifpn_aspp_regnetx_edgeailite(model_config, pretrained, base_model_class=RegNetX3p2GFMI4)


bifpn_aspp_regnetx3p2gf_bgr_edgeailite = bifpn_aspp_regnetx3p2gf_edgeailite
