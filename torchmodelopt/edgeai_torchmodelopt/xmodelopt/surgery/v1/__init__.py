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

import torch
import warnings
import torch
import warnings

from torchvision.ops.misc import SqueezeExcitation
from ....xnn import utils
from .replace_modules import replace_modules as replace_modules_func

from . import convert_to_lite 

default_replacement_flag_dict: dict[str, bool|dict] ={
    'squeeze_and_excite_to_identity' : True,
    'all_activation_to_relu': True,
    'relu_inplace_to_relu' : True,
    'gelu_to_relu' : True,
    'relu6_to_relu' : True,
    'silu_to_relu' : True,
    'hardswish_to_relu' : True,
    'hardsigmoid_to_relu' : True,
    'leakyrelu_to_relu' : True,
    'dropout_inplace_to_dropout':True,
    'conv2d_to_conv2d_dw_conv2d':True,
    'instancenorm_to_batchnorm':True,
    'groupnorm_to_batchnorm':False,
    'remove_identity': True
    # 'custom_surgery_flag':{},
}

flag_to_dict_entries:dict[str, dict] = {
    'squeeze_and_excite_to_identity':{SqueezeExcitation: [torch.nn.Identity]},
    'relu_inplace_to_relu':{torch.nn.ReLU: [torch.nn.ReLU]}, #'inplace' is not used
    'dropout_inplace_to_dropout':{torch.nn.Dropout: [torch.nn.Dropout, 'p']}, #'inplace' is not used
    'relu6_to_relu':{torch.nn.ReLU6: [torch.nn.ReLU]}, #'inplace' is not used
    'hardswish_to_relu':{torch.nn.Hardswish: [torch.nn.ReLU]}, #'inplace' is not used
    'hardsigmoid_to_relu':{torch.nn.Hardsigmoid: [torch.nn.ReLU]}, #'inplace' is not used
    'gelu_to_relu':{torch.nn.GELU: [torch.nn.ReLU]}, #'inplace' is not used
    'silu_to_relu':{torch.nn.SiLU: [torch.nn.ReLU]}, #'inplace' is not used
    'leakyrelu_to_relu':{torch.nn.LeakyReLU: [torch.nn.ReLU]},  # 'inplace' is not used
    'groupnorm_to_batchnorm':{torch.nn.GroupNorm: [convert_to_lite._replace_groupnorm]},
    'instancenorm_to_batchnorm':{torch.nn.InstanceNorm2d: [torch.nn.BatchNorm2d, 'num_features']},
    # with_normalization: whether to insert BN after replacing 3x3/5x5 conv etc. with dw-seperable conv
    # with_activation: whether to insert ReLU after replacing conv with dw-seperable conv
    'conv2d_to_conv2d_dw_conv2d':{torch.nn.Conv2d: [convert_to_lite._replace_conv2d, dict(groups_dw=None, group_size_dw=None, with_normalization=(True,False), with_activation=(True,False))]},
    # just a dummy entry to show that the key and value can be a functions
    # the key should return a boolean and the first entry of value(list) should return an instance of torch.nn.Module
    'remove_identity':{convert_to_lite._check_dummy: [convert_to_lite._replace_dummy]}
}


def _get_replacement_dict(replacement_flag_dict=None):
    replacement_flag_dict =replacement_flag_dict or get_replacement_flag_dict_default()
    replacement_dict = {}
    for k,v in replacement_flag_dict.items():
        if k in flag_to_dict_entries and v in (True,False):
                if v:
                    v = flag_to_dict_entries[k]
                else:
                    continue
        else:
            if not isinstance(v,dict):
                warnings.warn(f'if {k} is not a default flag or its value is not a boolean, the value must be a dict. So, this entry will be discarded!')
                continue
        replacement_dict.update(v)


def get_replacement_flag_dict_default(groups_dw=None, group_size_dw=None, **kwargs):
    '''
    A dictionary with the fllowing structure.
    key: a torch.nn.Module that has to be replaced OR a callable which takes a module as input and returns boolean
    value: a list. the fist entry is a constructor or a callable that creates the replacement module
                the remaining entries are properties that have to be copied from old module to newly created module.
    '''
    default_replacement_flag_dict['conv2d_to_conv2d_dw_conv2d'][torch.nn.Conv2d][1].update(groups_dw=groups_dw, group_size_dw=group_size_dw)
    return default_replacement_flag_dict


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
    replacement_dict = replacement_dict or get_replacement_flag_dict_default(**kwargs)
    replacement_dict = _get_replacement_dict(replacement_dict)
    model = replace_modules_func(model, inplace=inplace, replacement_dict=replacement_dict)
    return model


