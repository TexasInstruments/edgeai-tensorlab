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

import copy
import functools
import torch
from ....ops.misc import SqueezeExcitation
from .. import utils
from .. import layers


__all__ = {'convert_to_lite_model', 'create_lite_model'}


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


# this function can be used after creating the model to transform it into a lite model.
def create_lite_model(model_function, pretrained_backbone_names=None, pretrained=None, model_urls_dict=None,
                      model_name_lite=None, **kwargs):
    model_name_lite = model_name_lite or f'{model_function.__name__}_lite'
    lookup_pretrained = pretrained is True and model_urls_dict is not None and model_name_lite in model_urls_dict
    pretrained = model_urls_dict[model_name_lite] if lookup_pretrained else pretrained
    model = create_lite_model_impl(model_function, pretrained_backbone_names, pretrained=pretrained, **kwargs)
    return model


def create_lite_model_impl(model_function, pretrained_backbone_names=None, groups_dw=None, group_size_dw=1,
                           with_normalization=(True,False), with_activation=(True,False), **kwargs):
    pretrained = kwargs.pop('pretrained', None)
    pretrained_backbone = kwargs.pop('pretrained_backbone', None)
    # if pretrained is set to true, we will try to hanlde it inside model
    model = model_function(pretrained=(pretrained is True), pretrained_backbone=(pretrained_backbone is True), **kwargs)
    model = convert_to_lite_model(model, groups_dw=groups_dw, group_size_dw=group_size_dw,
                                  with_normalization=with_normalization, with_activation=with_activation)
    if pretrained and pretrained is not True:
        utils.load_weights(model, pretrained, state_dict_name=['state_dict', 'model'])
    elif pretrained_backbone and pretrained_backbone is not True:
        pretrained_backbone_names = pretrained_backbone_names if pretrained_backbone_names else {'^features.':'backbone.'}
        utils.load_weights(model, pretrained_backbone, state_dict_name=['state_dict', 'model'],
                           change_names_dict=pretrained_backbone_names)
    #
    return model


def convert_to_lite_model(model, inplace=True, replace_se=True, replace_conv=True, groups_dw=None, group_size_dw=1,
                          with_normalization=(True,False), with_activation=(True,False)):
    '''
    with_normalization: whether to insert BN after replacing 3x3/5x5 conv etc. with dw-seperable conv
    with_activation: whether to insert ReLU after replacing conv with dw-seperable conv
    '''
    model = model if inplace else copy.deepcopy(inplace)
    for name, m in model.named_modules():
        if replace_se:
            _replace_se(model, m)
        #
    #
    for name, m in model.named_modules():
        if replace_conv:
            _replace_conv2d(model, m, groups_dw=groups_dw, group_size_dw=group_size_dw,
                            with_normalization=with_normalization, with_activation=with_activation)
        #
    #
    for name, m in model.named_modules():
        _replace_with_new_module(model, m)
    #
    return model


_REPLACEMENTS_DICT = {
    torch.nn.ReLU6: [torch.nn.ReLU, 'inplace'],
    torch.nn.Hardswish: [torch.nn.ReLU, 'inplace'],
    torch.nn.SiLU: [torch.nn.ReLU, 'inplace'],
}


def _replace_module(model, m, new_m):
    _initialize_module(new_m)
    parent = utils.get_parent_module(model, m)
    name = utils.get_module_name(parent, m)
    new_m.train(model.training)
    setattr(parent, name, new_m)


def _get_replacement(m, relacements_dict=_REPLACEMENTS_DICT):
    for k_cls, new_cls_params in relacements_dict.items():
        if isinstance(m, k_cls):
            return new_cls_params
        #
    #
    return None


def _replace_with_new_module(model, m, new_cls_params=None, relacements_dict=_REPLACEMENTS_DICT):
    if new_cls_params is None:
        new_cls_params = _get_replacement(m, relacements_dict=relacements_dict)
    #
    if new_cls_params:
        new_cls = new_cls_params[0]
        new_args = {}
        for k in new_cls_params[1:]:
            new_args.update({k:getattr(m,k)})
        #
        new_m = new_cls(new_args)
        _replace_module(model, m, new_m)
    #


def _is_se_layer(m):
    if isinstance(m, functools.partial):
        m = m.func
    #
    return isinstance(m, SqueezeExcitation)


def _replace_se(model, m):
    if _is_se_layer(m):
        new_cls_params = [torch.nn.Identity]
        _replace_with_new_module(model, m, new_cls_params)


def _replace_conv2d(model, m, groups_dw=None, group_size_dw=1,
                    with_normalization=(True,False), with_activation=(True,False)):
    '''replace a regular convolution such as 3x3 or 5x5 with depthwise separable.
    with_normalization: introduce a normaliztion after point convolution default: (True,False)
    with_activation: introduce an activation fn between the depthwise and pointwise default: (True,False).
    - the above default choices are done to mimic a regulr convolution at a much lower complexity.
    - while replacing a regular convolution with a dws, we may not want to put a bn at the end,
      as a bn may be already there after this conv. same case with activation
    '''
    if isinstance(m, torch.nn.Conv2d):
        kernel_size = m.kernel_size if isinstance(m.kernel_size, (list,tuple)) else (m.kernel_size,m.kernel_size)
        if m.groups == 1 and m.in_channels >= 16 and kernel_size[0] > 1 and kernel_size[1] > 1:
            with_bias = m.bias is not None
            normalization = (m.with_normalization[0],m.with_normalization[1]) if hasattr(m, "with_normalization") else \
                (with_normalization[0],with_normalization[1])
            activation = (m.with_activation[0],m.with_activation[1]) if hasattr(m, "with_activation") else \
                (with_activation[0],with_activation[1])
            new_m = layers.ConvDWSepNormAct2d(in_planes=m.in_channels, out_planes=m.out_channels,
                        kernel_size=m.kernel_size, stride=m.stride, groups_dw=groups_dw, group_size_dw=group_size_dw,
                        bias=with_bias, normalization=normalization, activation=activation)
            _replace_module(model, m, new_m)
        #


def _initialize_module(module):
    _initialize_module_impl(module)
    for m in module.modules():
        _initialize_module_impl(m)
    #


def _initialize_module_impl(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Linear):
        init_range = 1.0 / math.sqrt(m.out_features)
        torch.nn.init.uniform_(m.weight, -init_range, init_range)
        torch.nn.init.zeros_(m.bias)
    #
