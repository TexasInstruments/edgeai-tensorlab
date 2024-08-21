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

import enum
import torch
from torch import nn
import types
from types import FunctionType,BuiltinFunctionType
from typing import Union, Dict, Any
import warnings
from functools import partial
from copy import deepcopy

from . import custom_modules, custom_surgery_functions,surgery
from .surgery import SurgeryModule, _replace_unsupported_layers, get_replacement_dict_default

# for repo specific modules 
try:
    from timm.layers.squeeze_excite import SEModule
except:
    # this will make the program skip the search for pattern
    SEModule = None


def convert_to_lite_fx(model:torch.nn.Module, example_input:list=[], example_kwargs:dict={}, replacement_dict:Dict[Any,Union[torch.nn.Module,callable]]=None, aten_graph:bool = False, verbose_mode:bool=False, *args, **kwargs):
    return replace_unsupported_layers(model, example_input= example_input, example_kwargs=example_kwargs, replacement_dict=replacement_dict, aten_graph=aten_graph, verbose_mode=verbose_mode, **kwargs)


#Flags for replacement dict
# for custom 
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
    'break_maxpool2d_with_kernel_size_greater_than_equalto_5':True,
    'break_avgpool2d_with_kernel_size_greater_than_equalto_5':True,
    'convert_resize_params_size_to_scale':True,
    'promote_conv2d_with_even_kernel_to_larger_odd_kernel':False,
    'break_conv2d_with_kernel_size_greater_than_7':False,
    # 'custom_surgery_flag':{},
}


default_replacement_flag_dict_no_training:dict[str,bool|dict] ={
    'relu6_to_relu' : True,
    'break_maxpool2d_with_kernel_size_greater_than_equalto_5':True,
    'break_avgpool2d_with_kernel_size_greater_than_equalto_5':True,
    'convert_resize_params_size_to_scale':True,
    'promote_conv2d_with_even_kernel_to_larger_odd_kernel':False,
    # 'custom_surgery_flag':{},
}


string_to_dict_entries:dict [str:dict] ={
    'squeeze_and_excite_to_identity' : {SEModule:nn.Identity,},
    'all_activation_to_ReLu': {nn.ReLU:nn.ReLU, nn.ReLU6:nn.ReLU, nn.GELU:nn.ReLU, nn.SiLU:nn.ReLU, nn.Hardswish:nn.ReLU, nn.Hardsigmoid:nn.ReLU, nn.LeakyReLU:nn.ReLU,},
    'relu_inplace_to_relu' : {nn.ReLU: nn.ReLU},
    'gelu_to_relu' : {nn.GELU: nn.ReLU},
    'relu6_to_relu' : {nn.ReLU6: nn.ReLU},
    'silu_to_relu' : {nn.SiLU: nn.ReLU},
    'hardswish_to_relu' : {nn.Hardswish: nn.ReLU},
    'hardsigmoid_to_relu' : {nn.Hardsigmoid: nn.ReLU},
    'leakyrelu_to_relu' : {nn.LeakyReLU: nn.ReLU},
    'dropout_inplace_to_dropout':{nn.Dropout: nn.Dropout},
    'promote_conv2d_with_even_kernel_to_larger_odd_kernel':{nn.Conv2d:custom_surgery_functions.gen_func_for_conv2d_even_kernel_to_odd},
    'break_conv2d_with_kernel_size_greater_than_7':{nn.Conv2d:custom_surgery_functions.gen_func_for_conv2d_kernel_gt_7},
    'break_maxpool2d_with_kernel_size_greater_than_equalto_5':{nn.MaxPool2d:custom_surgery_functions.gen_func_for_pool},
    'break_avgpool2d_with_kernel_size_greater_than_equalto_5':{nn.AvgPool2d:custom_surgery_functions.gen_func_for_pool},
    'convert_resize_params_size_to_scale':{nn.Upsample:custom_surgery_functions.gen_func_for_upsample},
}

def get_replacement_dict(
    replacement_flag_dict: dict[str|nn.Module|FunctionType|type,bool|nn.Module|FunctionType|type|tuple[FunctionType,FunctionType]]=None,
    can_retrain:bool = True
    ):
    
    if can_retrain:
        replacement_flag_dict = replacement_flag_dict or default_replacement_flag_dict
    else:
        replacement_flag_dict = replacement_flag_dict or default_replacement_flag_dict_no_training
        
    replacement_dict:dict[Any,list[tuple]] = {}
    
    for k,v in replacement_flag_dict.items():
        if k in string_to_dict_entries and v in (True,False):
                if v:
                    v = string_to_dict_entries[k]
                else:
                    continue
        else:
            if not isinstance(v,dict):
                warnings.warn(f'if {k} is not a default flag or its value is not a boolean, the value must be a dict. So, this entry will be discarded!')
                continue
        
        def adjust_value_for_replacement_dict(v1):
            input_adjustment_func = None
            if isinstance(v1,type) and issubclass(v1,nn.Module):
                v1 = v1()
                
            if isinstance(v1,dict):
                assert 'func' in v1 and isinstance(v1['func'],FunctionType) 
                assert 'kwargs' in v1 and isinstance (v1['kwargs'],dict)
                input_adjustment_func = v1.get('input_adjustment_func',None)
                v1 = partial(v1['func'],**v1['kwargs'])
            
            if not isinstance(v1,tuple):
                v1 = v1,input_adjustment_func
            else:
                assert len(v1) == 2
                v1 = adjust_value_for_replacement_dict(v1[0])[0],v1[1]
            
            return v1
        
        for k1,v1 in v.items():
            if isinstance(k1,nn.Module):
                k1 = type(k1)
            if isinstance(v1,list):
                v1 = [adjust_value_for_replacement_dict(item) for item in v1]
            else:
                v1 = adjust_value_for_replacement_dict(v1)
            
            if k1 in replacement_dict:
                if isinstance(v1 ,list):
                    replacement_dict[k1].extend(v1)
                else:
                    replacement_dict[k1].append(v1)
            else:
                replacement_dict[k1] = v1 if isinstance(v1,list) else [v1]
    
    return replacement_dict


# returns default dictionary for replacement
def get_replacement_dict_default():
    return default_replacement_flag_dict


def replace_unsupported_layers(model:nn.Module, example_input:list=[], example_kwargs:dict={}, replacement_dict:Dict[Any,Union[nn.Module,callable]]=None, aten_graph:bool = False, copy_args:list=[],  can_retrain=True, verbose_mode:bool=False):
    #TODO write appropiate documentation for this function
    
    '''
    main function that does the surgery
    key             |   value
    module/type     |   module (if same module is applicable all instances of former)
    module/type     |   type (if the __init__ doesn't require any positional arguement
                        and same  module with default keyword arguments (if any) is applicable
                        all instances of former )
    module/type     |   a replacement module generator function (generates a module based on
                        partition and main model or returns None if no replacement required) 
    module/type     |   tuple of two elements
                        first   : replacement module generator function (sane as prev) and 
                        second  : a input adjustment function based on the partition and inputs given to it (default: pass them as they appear in partition.input_nodes)
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
        self.replacement_dict=replacement_dict or get_replacement_dict_default()
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