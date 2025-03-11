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

import warnings
import torch
from torch import nn

try:
    from torchvision.ops.misc import SqueezeExcitation
    has_tv = True
except:
    has_tv = False

from .replace_modules import replace_modules as replace_modules_func
from . import convert_to_lite 
from ....xnn import utils
from ...utils.optimization_base import OptimizationBaseModule
from ...utils.transformation_utils import wrapped_transformation_fn

def convert_to_lite_model(model, replacement_dict=None, inplace=True, **kwargs):
    '''
    converts the model to lite model using replacement dict 
    wrapper to the function that does the surgery

    it does default surgery if no replacement dictionry is given
    replacement dictionry must contains flag name as keys and True/False or a replacement dictionary corresponding to flag as value
    
    behavior for each value:
    value               behavior
    True            ->  convert according to mapped replacement dict
    False           ->  discard it
    dict            ->  update the main replacement dictionary with the entries of it
    
    values for replacement dict
    key: a torch.nn.Module that has to be replaced OR a callable which takes a module as input and returns boolean
    value: a list. the fist entry is a constructor or a callable that creates the replacement module
                the remaining entries are properties that have to be copied from old module to newly created module.
    '''
    
    warnings.warn("WARNING - xmodelopt.v1.surgery can only replace modules. To replace functions or operators, please use the torch.fx based xmodelopt.v2.surgery instead")
    replacement_dict = replacement_dict or get_replacement_dict_default(**kwargs)
    replacement_dict = _get_replacement_dict(replacement_dict)
    model = replace_modules_func(model, inplace=inplace, replacement_dict=replacement_dict)
    return model


#Default Flags for replacement dict 
# for custom replacement add a custom flag name (any string not in default flag) as key and map it to a dict containing pattern and replacement
# note if same key is used for two pattern the last replacement will be performed
default_replacement_flag_dict = {
    'squeeze_and_excite_to_identity': True,
    'all_activation_to_relu'        : False,
    'relu_inplace_to_relu'          : True,
    'gelu_to_relu'                  : True,
    'relu6_to_relu'                 : True,
    'silu_to_relu'                  : True,
    'hardswish_to_relu'             : True,
    'hardsigmoid_to_relu'           : True,
    'leakyrelu_to_relu'             : True,
    'dropout_inplace_to_dropout'    : True,
    'conv2d_to_conv2d_dw_conv2d'    : True,
    'break_maxpool2d_with_kernel_\
        size_greater_than_equalto_5': False,
    'instancenorm_to_batchnorm'     : False,
    'groupnorm_to_batchnorm'        : False,
    'remove_identity'               : True
    # 'custom_surgery_flag':{},
}

#Mapping between the flags and the actual replacements corresponding to them
# This dictionary is used whenever a flag is enabled to fetch the corresponding replacement entries
flag_to_dict_entries = {
    'all_activation_to_relu'        : {nn.ReLU : nn.ReLU, 
                                        nn.ReLU6 : nn.ReLU, 
                                        nn.GELU : nn.ReLU,
                                        nn.SiLU : nn.ReLU, 
                                        nn.Hardswish : nn.ReLU, 
                                        nn.Hardsigmoid : nn.ReLU, 
                                        nn.LeakyReLU : nn.ReLU,},
    'relu_inplace_to_relu'          : {torch.nn.ReLU : [torch.nn.ReLU]}, #'inplace' is not used
    'dropout_inplace_to_dropout'    : {torch.nn.Dropout : [torch.nn.Dropout, 'p']}, #'inplace' is not used
    'relu6_to_relu'                 : {torch.nn.ReLU6 : [torch.nn.ReLU]}, #'inplace' is not used
    'hardswish_to_relu'             : {torch.nn.Hardswish: [torch.nn.ReLU]}, #'inplace' is not used
    'hardsigmoid_to_relu'           : {torch.nn.Hardsigmoid: [torch.nn.ReLU]}, #'inplace' is not used
    'gelu_to_relu'                  : {torch.nn.GELU: [torch.nn.ReLU]}, #'inplace' is not used
    'silu_to_relu'                  : {torch.nn.SiLU: [torch.nn.ReLU]}, #'inplace' is not used
    'leakyrelu_to_relu'             : {torch.nn.LeakyReLU: [torch.nn.ReLU]},  # 'inplace' is not used
    'groupnorm_to_batchnorm'        : {torch.nn.GroupNorm: [convert_to_lite._replace_group_norm]},
    'instancenorm_to_batchnorm'     : {torch.nn.InstanceNorm2d: [torch.nn.BatchNorm2d, 'num_features']},
    # with_normalization: whether to insert BN after replacing 3x3/5x5 conv etc. with dw-seperable conv
    # with_activation: whether to insert ReLU after replacing conv with dw-seperable conv
    'conv2d_to_conv2d_dw_conv2d'    : {torch.nn.Conv2d: [convert_to_lite._replace_conv2d, \
        dict(groups_dw=None, group_size_dw=None, with_normalization=(True,False), with_activation=(True,False))]},
    'break_maxpool2d_with_kernel_\
        size_greater_than_equalto_5': {torch.nn.MaxPool2d : [convert_to_lite.replace_maxpool2d]},
    # just a dummy entry to show that the key and value can be a functions
    # the key should return a boolean and the first entry of value(list) should return an instance of torch.nn.Module
    'remove_identity'               : {convert_to_lite._check_dummy: [convert_to_lite._replace_dummy]}
}
flag_to_dict_entries.update(squeeze_and_excite_to_identity=({SqueezeExcitation: [torch.nn.Identity]} if has_tv else {}))


def get_replacement_dict_default(groups_dw=None, group_size_dw=None, return_flags=True, **kwargs):
    '''
    when return_flags is True, returns default flag dictionary with the fllowing structure.
        key: a flag string
        value: True/False if it is registered in 'flag_to_dict_entries' else a dictionary containing the replacement entries corresponding to it
    otherwise, returns flag_to_dict_entries containing the following type of key/value pairs
        key: a string
        value: a dict with keys as module types to be replaced and values as the replacement functions/modules
    '''
    flag_to_dict_entries['conv2d_to_conv2d_dw_conv2d'][torch.nn.Conv2d][1].update(groups_dw=groups_dw, group_size_dw=group_size_dw)
    if return_flags:
        ret_val = default_replacement_flag_dict
    else:
        ret_val = flag_to_dict_entries
    
    return ret_val


def _get_replacement_dict(replacement_flag_dict=None):
    '''
    this function actually converts the flags mapped to True to their corresponding replacements
    if the flags is not registered in 'flag_to_dict_entries', its value should be a dict of replacement and that will be updated in the dictionary
    '''
    replacement_flag_dict =replacement_flag_dict or get_replacement_dict_default()
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
    return replacement_dict


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


class SurgeryModule(OptimizationBaseModule):
    '''
    wrapper module  for performing surgery on module

    it will do default surgery on model if no replacement dictionary is passed 
    while initializing.
    '''
    
    def __init__(self, model, *args, replacement_dict=None, transformation_dict=None, copy_attrs=None, **kwargs) -> None:
        '''perform surgery on the model and creates a new model'''
        copy_attrs= copy_attrs or []
        super().__init__(model,*args, transformation_dict=transformation_dict, copy_attrs=copy_attrs, **kwargs)
        self.replacement_dict=replacement_dict or get_replacement_dict_default()
        self.module = wrapped_transformation_fn(convert_to_lite_model, model, replacement_dict=replacement_dict,)

    def forward(self,x,*args,**kwargs):
        '''
        atleast one input required 
        for more input, add them as a part of args
        '''
        return self.module(x,*args,**kwargs)

    def get_replacement_dict(self):
        '''returns the default replacement dictionary that can be updated further'''
        return self.replacement_dict