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

from edgeai_torchmodelopt import xnn

from . import mobilenetv2
from . import mobilenetv1
from . import resnet
from . import shufflenetv2
from . import regnet
from . import vgg
from . import mobilenetv3

try: from .. import mobilenetv2_internal
except: pass

try: from .. import mobilenetv2_densenas_internal
except: pass

try: from .. import mobilenetv2_ericsun_internal
except: pass

try: from .. import mobilenetv2_shicai_internal
except: pass

try: from .. import flownetbase_internal
except: pass

try: from .. import mobilenetv1_internal
except: pass

from . import pixel2pixel

__all__ = ['regnetx400mf_x1', 'regnetx800mf_x1', 'regnetx1p6gf_x1', 'regnetx3p2gf_x1',
           'regnetx400mf_x1_bgr', 'regnetx800mf_x1_bgr', 'regnetx1p6gf_x1_bgr', 'regnetx3p2gf_x1_bgr',
           'mobilenetv1_x1',
           'mobilenetv2_tv_x1', 'mobilenetv2_x1', 'mobilenetv2_tv_x2_t2',
           'mobilenetv3_lite_small_x1', 'mobilenetv3_lite_large_x1', 'mobilenetv3_lite_large_x2r',
           'resnet50_x1', 'resnet50_xp5', 'resnet18_x1',
           'vgg16_x1',
           'shufflenetv2_x1p0', 'shufflenetv2_x1p5',
           # experimental
           'mobilenetv2_ericsun_x1', 'mobilenetv2_shicai_x1',
           'mobilenetv2_tv_gws_x1', 'mobilenetv1_multi_label_x1',
           'mobilenetv2_tv_dense_nas_x1'
]


#####################################################################
def resnet50_x1(model_config=None, pretrained=None, change_names_dict=None):
    model_config = resnet.get_config().merge_from(model_config)
    model = resnet.resnet50_with_model_config(model_config)
    # the pretrained model provided by torchvision and what is defined here differs slightly
    # note: that this change_names_dict  will take effect only if the direct load fails
    if change_names_dict is None:
        change_names_dict = {'^conv1.': 'features.conv1.', '^bn1.': 'features.bn1.',
                             '^relu.': 'features.relu.', '^maxpool.': 'features.maxpool.',
                             '^layer': 'features.layer', '^fc.': 'classifier.'}
    #
    if pretrained:
        model = xnn.utils.load_weights(model, pretrained, change_names_dict=change_names_dict)
    return model, change_names_dict


def resnet50_xp5(model_config=None, pretrained=None):
    model_config.width_mult = 0.5
    return resnet50_x1(model_config=model_config, pretrained=pretrained)


#####################################################################
def resnet18_x1(model_config=None, pretrained=None, change_names_dict=None):
    model_config = resnet.get_config().merge_from(model_config)
    model = resnet.resnet18_with_model_config(model_config)
    # the pretrained model provided by torchvision and what is defined here differs slightly
    # note: that this change_names_dict  will take effect only if the direct load fails
    if change_names_dict is None:
        change_names_dict = {'^conv1.': 'features.conv1.', '^bn1.': 'features.bn1.',
                             '^relu.': 'features.relu.', '^maxpool.': 'features.maxpool.',
                             '^layer': 'features.layer', '^fc.': 'classifier.'}
    #
    if pretrained:
        model = xnn.utils.load_weights(model, pretrained, change_names_dict=change_names_dict)
    return model, change_names_dict


#####################################################################
def mobilenetv1_x1(model_config=None, pretrained=None):
    model_config = mobilenetv1.get_config().merge_from(model_config)
    model = mobilenetv1.MobileNetV1(model_config=model_config)
    if pretrained:
        model = xnn.utils.load_weights(model, pretrained)
    return model

def mobilenetv1_multi_label_x1(model_config=None, pretrained=None):
    model_config = mobilenetv1.get_config().merge_from(model_config)
    model = mobilenetv1_internal.MobileNetV1MultiLabel(model_config=model_config)
    if pretrained:
        model = xnn.utils.load_weights(model, pretrained)
    return model


#####################################################################
def mobilenetv2_tv_x1(model_config=None, pretrained=None):
    model_config = mobilenetv2.get_config().merge_from(model_config)
    model = mobilenetv2.MobileNetV2TV(model_config=model_config)
    if pretrained:
        model = xnn.utils.load_weights(model, pretrained)
    return model
#
#alias
mobilenetv2_x1 = mobilenetv2_tv_x1


def mobilenetv2_tv_x2_t2(model_config=None, pretrained=None):
    model_config = mobilenetv2.get_config().merge_from(model_config)
    model_config.width_mult = 2.0
    model_config.expand_ratio = 2.0
    model = mobilenetv2.MobileNetV2TV(model_config=model_config)
    if pretrained:
        model = xnn.utils.load_weights(model, pretrained)
    return model


#####################################################################
def mobilenetv2_tv_gws_x1(model_config=None, pretrained=None):
    model_config = mobilenetv2_internal.get_config_mnetv2_gws().merge_from(model_config)
    model = mobilenetv2_internal.MobileNetV2TVGWS(model_config=model_config)
    if pretrained:
        model = xnn.utils.load_weights(model, pretrained)
    return model

def mobilenetv2_tv_dense_nas_x1(model_config, pretrained=None):
    model_config = mobilenetv2_densenas_internal.get_config_mnetv2_dense_nas().merge_from(model_config)
    model = mobilenetv2_densenas_internal.MobileNetV2TVDenseNAS(model_config=model_config)
    if pretrained:
        model = xnn.utils.load_weights(model, pretrained)
    return model


#####################################################################
def mobilenetv2_ericsun_x1(model_config=None, pretrained=None):
    model_config = mobilenetv2_ericsun_internal.get_config().merge_from(model_config)
    model = mobilenetv2_ericsun_internal.MobileNetV2Ericsun(model_config=model_config)
    if pretrained:
        model = xnn.utils.load_weights(model, pretrained)
    return model


def mobilenetv2_shicai_x1(model_config=None, pretrained=None):
    model_config = mobilenetv2_shicai_internal.get_config().merge_from(model_config)
    model = mobilenetv2_shicai_internal.mobilenetv2_shicai(model_config=model_config)
    if pretrained:
        model = xnn.utils.load_weights(model, pretrained)
    return model


#####################################################################
def shufflenetv2_x1p0(model_config=None, pretrained=None):
    return shufflenetv2.shufflenet_v2_x1_0(model_config=model_config, pretrained=pretrained)


def shufflenetv2_x1p5(model_config=None, pretrained=None):
    return shufflenetv2.shufflenet_v2_x1_5(model_config=model_config, pretrained=pretrained)


#####################################################################
# here this is nothing specific about bgr in this model
# but is just a reminder that regnet models are typically trained with bgr input
def vgg16_x1(model_config=None, pretrained=None, change_names_dict=None, **kwargs):
    #model_config = resnet.get_config().merge_from(model_config)
    model = vgg.vgg16(pretrained=pretrained, **kwargs)
    if pretrained:
        model = xnn.utils.load_weights(model, pretrained, change_names_dict=None,
                                       state_dict_name=['state_dict','model_state'])
    return model, change_names_dict


#####################################################################
def regnetx_base(model=None, pretrained=None, change_names_dict=None):
    # the pretrained model provided by torchvision and what is defined here differs slightly
    # note: that this change_names_dict  will take effect only if the direct load fails
    if change_names_dict is None:
        change_names_dict = {'^stem.': 'features.stem.',
                             '^s1': 'features.s1',
                             '^s2': 'features.s2',
                             '^s3': 'features.s3',
                             '^s4': 'features.s4'}
    #
    if pretrained:
        model = xnn.utils.load_weights(model, pretrained, change_names_dict=change_names_dict,
                                       state_dict_name=['state_dict','model_state'])
    return model, change_names_dict


def regnetx400mf_x1(model_config=None, pretrained=None, change_names_dict=None):
    model_config = regnet.get_config().merge_from(model_config)
    model = regnet.regnetx400mf_with_model_config(model_config)
    return regnetx_base(model, pretrained, change_names_dict)

regnetx400mf_x1_bgr = regnetx400mf_x1


def regnetx800mf_x1(model_config=None, pretrained=None, change_names_dict=None):
    model_config = regnet.get_config().merge_from(model_config)
    model = regnet.regnetx800mf_with_model_config(model_config)
    return regnetx_base(model, pretrained, change_names_dict)

regnetx800mf_x1_bgr = regnetx800mf_x1


def regnetx1p6gf_x1(model_config=None, pretrained=None, change_names_dict=None):
    model_config = regnet.get_config().merge_from(model_config)
    model = regnet.regnetx1p6gf_with_model_config(model_config)
    return regnetx_base(model, pretrained, change_names_dict)

regnetx1p6gf_x1_bgr = regnetx1p6gf_x1


def regnetx3p2gf_x1(model_config=None, pretrained=None, change_names_dict=None):
    model_config = regnet.get_config().merge_from(model_config)
    model = regnet.regnetx3p2gf_with_model_config(model_config)
    return regnetx_base(model, pretrained, change_names_dict)

regnetx3p2gf_x1_bgr = regnetx3p2gf_x1


#####################################################################
def mobilenetv3_small_x1(model_config=None, pretrained=None):
    model_config = mobilenetv3.get_config().merge_from(model_config)
    model = mobilenetv3.mobilenet_v3_small(model_config=model_config)
    if pretrained:
        model = xnn.utils.load_weights(model, pretrained)
    return model


def mobilenetv3_large_x1(model_config=None, pretrained=None):
    model_config = mobilenetv3.get_config().merge_from(model_config)
    model = mobilenetv3.mobilenet_v3_large(model_config=model_config)
    if pretrained:
        model = xnn.utils.load_weights(model, pretrained)
    return model


def mobilenetv3_lite_small_x1(model_config=None, pretrained=None):
    model_config = mobilenetv3.get_config().merge_from(model_config)
    model = mobilenetv3.mobilenet_v3_lite_small(model_config=model_config)
    if pretrained:
        model = xnn.utils.load_weights(model, pretrained)
    return model


def mobilenetv3_lite_large_x1(model_config=None, pretrained=None):
    model_config = mobilenetv3.get_config().merge_from(model_config)
    model = mobilenetv3.mobilenet_v3_lite_large(model_config=model_config)
    if pretrained:
        model = xnn.utils.load_weights(model, pretrained)
    return model


def mobilenetv3_lite_large_x2r(model_config=None, pretrained=None):
    '''double the number of channels of mobilenetv3_lite_large_x1
     _reduced_tail = True to prevent the number of channels from being too large at the end'''
    model_config = mobilenetv3.get_config().merge_from(model_config)
    model_config.width_mult = 2.0
    model = mobilenetv3.mobilenet_v3_lite_large(model_config=model_config, _reduced_tail=True)
    if pretrained:
        model = xnn.utils.load_weights(model, pretrained)
    return model
