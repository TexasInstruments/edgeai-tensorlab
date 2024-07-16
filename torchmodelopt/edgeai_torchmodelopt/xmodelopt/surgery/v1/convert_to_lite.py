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

import copy
import functools
import torch
import inspect
import math
import warnings

from torchvision.ops.misc import SqueezeExcitation
from ....xnn import utils
from ....xnn import layers
from .replace_modules import replace_modules as replace_modules_func


#################################################################################################
# # Note: currently the creation of lite models follows the following syntax:
# def mobilenet_v3_large_lite( **kwargs):
#     return xnn.model_surgery.create_lite_model(mobilenet_v3_large, **kwargs)
# # i.e., the model sugery is handled after the model creation is completely over.
# #
# # yet another potential option will be to handle model_sugery inside the model
# # and then to pass a suitable model_surgery function from outside
# # this will need model_surgery to be handled n every model,
# # so not using this method for the time being
# def mobilenet_v3_large_lite( **kwargs):
#     return mobilenet_v3_large(model_surgery=xnn.model_surgery.convert_to_lite_model, **kwargs)
#################################################################################################


__all__ = ['convert_to_lite_model', 'create_lite_model', 'get_replacement_dict_default']


def _check_dummy(current_m):
    '''
    a dummy check function to demonstrate a posible key in replacements dict.
    replace with your condition check.
    '''
    return isinstance(current_m, torch.nn.Identity)


def _replace_dummy(current_m):
    '''
    a dummy replace function to demonstrate a posible value in replacements dict.
    replace with the new object that you wish to replace
    Note: if the output is same as input, no replacement is done.
    '''
    return current_m


def _replace_conv2d(current_m=None, groups_dw=None, group_size_dw=None,
                    with_normalization=(True,False), with_activation=(True,False)):
    '''replace a regular convolution such as 3x3 or 5x5 with depthwise separable.
    with_normalization: introduce a normaliztion after point convolution default: (True,False)
    with_activation: introduce an activation fn between the depthwise and pointwise default: (True,False).
    - the above default choices are done to mimic a regulr convolution at a much lower complexity.
    - while replacing a regular convolution with a dws, we may not want to put a bn at the end,
      as a bn may be already there after this conv. same case with activation

    Note: it is also possible to do checks inside this function and if we dont want to replace, return the original module
    '''
    assert current_m is not None, 'for replacing Conv2d the current module must be provided'
    if isinstance(current_m, torch.nn.Conv2d) and (groups_dw is not None or group_size_dw is not None):
        kernel_size = current_m.kernel_size if isinstance(current_m.kernel_size, (list,tuple)) else (current_m.kernel_size,current_m.kernel_size)
        if (current_m.groups == 1 and current_m.in_channels >= 16 and kernel_size[0] > 1 and kernel_size[1] > 1):
            with_bias = current_m.bias is not None
            normalization = (current_m.with_normalization[0],current_m.with_normalization[1]) if hasattr(current_m, "with_normalization") else \
                (with_normalization[0],with_normalization[1])
            activation = (current_m.with_activation[0],current_m.with_activation[1]) if hasattr(current_m, "with_activation") else \
                (with_activation[0],with_activation[1])
            new_m = layers.ConvDWSepNormAct2d(in_planes=current_m.in_channels, out_planes=current_m.out_channels,
                        kernel_size=current_m.kernel_size, stride=current_m.stride, groups_dw=groups_dw, group_size_dw=group_size_dw,
                        bias=with_bias, normalization=normalization, activation=activation)
            return new_m
        #
    #
    return current_m


def _replace_group_norm(current_m=None):
    assert current_m is not None, 'for replacing GroupNorm the current module must be provided'
    if isinstance(current_m, torch.nn.GroupNorm):
        new_m = torch.nn.BatchNorm2d(num_features=current_m.num_channels)
        return new_m
    #
    return current_m


def get_replacement_dict_default(groups_dw=None, group_size_dw=None, replace_group_norm=False, replace_instance_norm=False, **kwargs):
    '''
    A dictionary with the fllowing structure.
    key: a torch.nn.Module that has to be replaced OR a callable which takes a module as input and returns boolean
    value: a list. the fist entry is a constructor or a callable that creates the replacement module
                the remaining entries are properties that have to be copied from old module to newly created module.
    '''
    replacement_dict_lite = {
        torch.nn.ReLU: [torch.nn.ReLU], #'inplace' is not used
        torch.nn.Dropout: [torch.nn.Dropout, 'p'], #'inplace' is not used
        torch.nn.ReLU6: [torch.nn.ReLU], #'inplace' is not used
        torch.nn.Hardswish: [torch.nn.ReLU], #'inplace' is not used
        torch.nn.SiLU: [torch.nn.ReLU], #'inplace' is not used
        torch.nn.LeakyReLU: [torch.nn.ReLU],  # 'inplace' is not used
        SqueezeExcitation: [torch.nn.Identity],
        # with_normalization: whether to insert BN after replacing 3x3/5x5 conv etc. with dw-seperable conv
        # with_activation: whether to insert ReLU after replacing conv with dw-seperable conv
        torch.nn.Conv2d: [_replace_conv2d, dict(groups_dw=groups_dw, group_size_dw=group_size_dw, with_normalization=(True,False), with_activation=(True,False))],
        # just a dummy entry to show that the key and value can be a functions
        # the key should return a boolean and the first entry of value(list) should return an instance of torch.nn.Module
        _check_dummy: [_replace_dummy]
    }

    if replace_group_norm:
        replacement_dict_lite.update({
            torch.nn.GroupNorm: [_replace_group_norm]
        })

    if replace_instance_norm:
        replacement_dict_lite.update({
            torch.nn.InstanceNorm2d: [torch.nn.BatchNorm2d, 'num_features']
        })

    return replacement_dict_lite


# this function can be used after creating the model to transform it into a lite model.
def create_lite_model(model_function, pretrained_backbone_names=None, pretrained=None, model_urls_dict=None,
                      model_name_lite=None, replacement_dict=None, **kwargs):
    model_name_lite = model_name_lite or f'{model_function.__name__}_lite'
    lookup_pretrained = pretrained is True and model_urls_dict is not None and model_name_lite in model_urls_dict
    pretrained = model_urls_dict[model_name_lite] if lookup_pretrained else pretrained
    model = _create_lite_model_impl(model_function, pretrained_backbone_names, pretrained=pretrained,
                                   replacement_dict=replacement_dict, **kwargs)
    return model


def _create_lite_model_impl(model_function, pretrained_backbone_names=None, replacement_dict=None, **kwargs):
    pretrained = kwargs.pop('pretrained', None)
    pretrained_backbone = kwargs.pop('pretrained_backbone', None)
    # if pretrained is set to true, we will try to hanlde it inside model
    if pretrained_backbone is not None:
        model = model_function(pretrained=(pretrained is True), pretrained_backbone=(pretrained_backbone is True), **kwargs)
    else:
        model = model_function(pretrained=(pretrained is True), **kwargs)
    #
    model = convert_to_lite_model(model, replacement_dict=replacement_dict, **kwargs)
    if pretrained and pretrained is not True:
        utils.load_weights(model, pretrained, state_dict_name=['state_dict', 'model'])
    elif pretrained_backbone and pretrained_backbone is not True:
        pretrained_backbone_names = pretrained_backbone_names if pretrained_backbone_names else {'^features.':'backbone.'}
        utils.load_weights(model, pretrained_backbone, state_dict_name=['state_dict', 'model'],
                           change_names_dict=pretrained_backbone_names)
    #
    return model


def convert_to_lite_model(model, inplace=True, replacement_dict=None, **kwargs):
    warnings.warn("WARNING - xmodelopt.v1.surgery is based on the modules. For superior functionality, please use the torch.fx based xmodelopt.v2.surgery instead")
    replacement_dict = replacement_dict or get_replacement_dict_default(**kwargs)
    model = replace_modules_func(model, inplace=inplace, replacement_dict=replacement_dict)
    return model

