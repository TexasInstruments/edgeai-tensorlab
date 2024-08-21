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
from torch import nn
from copy import deepcopy
from types import FunctionType
from typing import Union, Dict, Any
import warnings

from . import custom_modules, custom_surgery_functions
from .surgery import _replace_unsupported_layers

from torchvision.ops.misc import SqueezeExcitation
try:
    from timm.layers.squeeze_excite import SEModule
except:
    SEModule = None


def convert_to_lite_fx(model:torch.nn.Module,replacement_dict:Dict[Any,Union[torch.nn.Module,callable]]=None, verbose_mode:bool=False, example_input = None,**kwargs):
    '''
    converts model into lite model using replacement dict
    if no replacement dict is provided it does the default replacement
    '''
    if example_input is None:
        warnings.warn("example_input optional and used only in models using LayerNorm. Using a default value since it was not provided.")
        example_input = torch.rand(1,3,224,224) # Default input shape
    return replace_unsupported_layers(model, replacement_dict=replacement_dict, verbose_mode=verbose_mode,example_input=example_input, **kwargs)


#Default Flags for replacement dict
# for custom replacement add a custom flag name (any string not in default flag) as key and map it to a dict containing pattern and replacement
# note if same key is used for two pattern the last replacement will be performed
default_replacement_flag_dict: dict[str, bool|dict] ={
    'squeeze_and_excite_to_identity' : True,
    'all_activation_to_relu': False,
    'relu_inplace_to_relu' : True,
    'gelu_to_relu' : True,
    'relu6_to_relu' : True,
    'silu_to_relu' : True,
    'hardswish_to_relu' : True,
    'hardsigmoid_to_relu' : True,
    'leakyrelu_to_relu' : True,
    'dropout_inplace_to_dropout':True,
    'replace_CNBlock':True,
    'focus_to_optimized_focus':True,
    'break_maxpool2d_with_kernel_size_greater_than_equalto_5':True,
    'break_avgpool2d_with_kernel_size_greater_than_equalto_5':True,
    'convert_resize_params_size_to_scale':True,
    'replace_conv_k_size_6_to_k_size_5':True,
    # 'promote_conv2d_with_even_kernel_to_larger_odd_kernel':False,
    'break_conv2d_with_kernel_size_greater_than_7':False,
    # 'custom_surgery_flag':{},
}

#Default Flags for replacement dict with no training required as subset of default flags
# for custom replacement add a custom flag name (any string not in default flag) as key and map it to a dict containing pattern and replacement
# note if same key is used for two pattern the last replacement will be performed
default_replacement_flag_dict_no_training:dict[str,bool|dict] ={
    'relu6_to_relu' : True,
    'dropout_inplace_to_dropout':True,
    'focus_to_optimized_focus':True,
    'break_maxpool2d_with_kernel_size_greater_than_equalto_5':True,
    'break_avgpool2d_with_kernel_size_greater_than_equalto_5':True,
    'convert_resize_params_size_to_scale':True,
    # 'promote_conv2d_with_even_kernel_to_larger_odd_kernel':False,
    # 'custom_surgery_flag':{},
}

#Mapping between the flags and the actual replacements corresponding to them
# This dictionary is used whenever a flag is enabled to fetch the corresponding replacement entries
flag_to_dict_entries:dict [str:dict] ={
    'squeeze_and_excite_to_identity' : {SEModule:nn.Identity,custom_modules.SEModule():nn.Identity(),custom_modules.SEModule1():nn.Identity(),'se_layer':custom_surgery_functions.replace_se_layer},
    'all_activation_to_ReLu': {nn.ReLU:nn.ReLU, nn.ReLU6:nn.ReLU, nn.GELU:nn.ReLU, nn.SiLU:nn.ReLU, nn.Hardswish:nn.ReLU, nn.Hardsigmoid:nn.ReLU, nn.LeakyReLU:nn.ReLU,},
    'relu_inplace_to_relu' : {nn.ReLU: nn.ReLU},
    'gelu_to_relu' : {nn.GELU: nn.ReLU},
    'relu6_to_relu' : {nn.ReLU6: nn.ReLU},
    'silu_to_relu' : {nn.SiLU: nn.ReLU},
    'hardswish_to_relu' : {nn.Hardswish: nn.ReLU},
    'hardsigmoid_to_relu' : {nn.Hardsigmoid: nn.ReLU},
    'leakyrelu_to_relu' : {nn.LeakyReLU: nn.ReLU},
    'dropout_inplace_to_dropout':{nn.Dropout: nn.Dropout},
    'replace_CNBlock':{'CNBlock':custom_surgery_functions.replace_cnblock},
    'focus_to_optimized_focus':{custom_modules.Focus():custom_modules.OptimizedFocus()},
    'replace_conv_k_size_6_to_k_size_5':{'conv_6':custom_surgery_functions.replace_conv2d_kernel_size_6},
    'break_conv2d_with_kernel_size_greater_than_7':{'conv_ge_7':custom_surgery_functions.replace_conv2d_kernel_size_gt_7},
    'break_maxpool2d_with_kernel_size_greater_than_equalto_5':{'maxpool_ge_5':custom_surgery_functions.replace_maxpool2d_kernel_size_ge_5},
    'break_avgpool2d_with_kernel_size_greater_than_equalto_5':{'avgpool_ge_5':custom_surgery_functions.replace_avgpool2d_kernel_size_ge_5},
    'convert_resize_params_size_to_scale':{'upsample':custom_surgery_functions.replace_resize_with_scale_factor},
}


# returns default dictionary for replacement
def get_replacement_flag_dict_default():
    '''
    returns the default flag dictionary.
    to see the dict print 'default_replacement_flag_dict' from the file this function is in
    '''
    return default_replacement_flag_dict


def get_replacement_dict(
    replacement_flag_dict: dict[str|nn.Module|FunctionType|type,bool|nn.Module|FunctionType|type|tuple[FunctionType,FunctionType]]=None,
    can_retrain:bool = True
    ):
    '''
    this function actually converts the flags mapped to True to their corresponding replacements
    if no replacement_flag_dict is given it uses default flag dictionary based on can_retrain
    if the flags is not registered in 'flag_to_dict_entries', its value should be a dict of replacement and that will be updated in the dictionary
    '''
    
    if can_retrain:
        replacement_flag_dict = replacement_flag_dict or default_replacement_flag_dict
    else:
        replacement_flag_dict = replacement_flag_dict or default_replacement_flag_dict_no_training
        
    replacement_dict:dict[Any,list[tuple]] = {}
    
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


def replace_unsupported_layers(model:nn.Module, example_input:list=[], example_kwargs:dict={}, replacement_dict:Dict[Any,Union[nn.Module,callable]]=None, aten_graph:bool = False, copy_args:list=[],  can_retrain=True, verbose_mode:bool=False):
    #TODO write appropiate documentation for this function
    
    '''
    wrapper to the function that does the surgery

    it does default surgery if no replacement dictionry is given
    replacement dictionry must contains flag name as keys and True/False or a replacement dictionary corresponding to flag as value
    
    behavior for each value:
    value               behavior
    True            ->  convert according to mapped replacement dict
    False           ->  discard it
    dict            ->  update the main replacement dictionary with the entries of it
    
    values for replacement dict
    keys                value
    callable        ->  callable            : any call function to call_function if they take same argument partial agument may -                                         be used 
    callable        ->  nn.Module           : any call function to call_function if they take same argument partial agument may -                                         be used 
    Any             ->  Callable            : any self-made surgery function 
    nn.Module       ->  nn.Module/type          : any nn.Module pattern to replace with another nn.Module
    type            ->  type/nn.Module      : replaces sub-module of same type as patttern using traditional python approach 
    '''
    
    #TODO Check for functions    
    if model.training:
        RuntimeWarning("The model is in train mode, converting to eval mode. This might change the network behaviour.")
        model.eval()
        is_train_mode = True
    else:
        is_train_mode = False
        
    replacement_dict = get_replacement_dict(replacement_dict,can_retrain=can_retrain)
    
    model = deepcopy(model)
    
    final_model = _replace_unsupported_layers(model,example_input,example_kwargs,replacement_dict,aten_graph,copy_args,verbose_mode)
    
    if is_train_mode:
        final_model.train()
    
    return final_model


class SurgeryModule(torch.nn.Module):
    '''
    wrapper module  for performing surgery on module

    it will do default surgery on model if no replacement dictionary is passed 
    while initializing.
    '''
    
    def __init__(self, model, replacement_dict=None) -> None:
        '''perform surgery on the model and creates a new model'''
        super().__init__()
        self.replacement_dict=replacement_dict or get_replacement_flag_dict_default()
        self.module = replace_unsupported_layers(model, self.replacement_dict)

    def forward(self,x,*args,**kwargs):
        '''
        atleast one input required 
        for more input, add them as a part of args
        '''
        return self.module(x,*args,**kwargs)

    def get_replacement_dict(self):
        '''returns the default replacement dictionary that can be updated further'''
        return self.replacement_dict