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
from torchvision.ops.misc import SqueezeExcitation
from copy import deepcopy

from . import custom_modules, custom_surgery_functions,surgery
from .surgery import _replace_unsupported_layers

# for repo specific modules 
try:
    from timm.layers.squeeze_excite import SEModule
except:
    # this will make the program skip the search for pattern
    SEModule = None


def convert_to_lite_fx(model:torch.nn.Module, example_input:list=[], example_kwargs:dict={}, replacement_dict:Dict[Any,Union[torch.nn.Module,callable]]=None, aten_graph:bool = False, verbose_mode:bool=False, *args, **kwargs):
    '''
    converts model into lite model using replacement dict
    if no replacement dict is provided it does the default replacement
    '''
    return replace_unsupported_layers(model, example_input= example_input, example_kwargs=example_kwargs, replacement_dict=replacement_dict, aten_graph=aten_graph, verbose_mode=verbose_mode, **kwargs)


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
    'break_maxpool2d_with_kernel_size_greater_than_equalto_5':True,
    'break_avgpool2d_with_kernel_size_greater_than_equalto_5':True,
    'convert_resize_params_size_to_scale':True,
    'promote_conv2d_with_even_kernel_to_larger_odd_kernel':False,
    'break_conv2d_with_kernel_size_greater_than_7':False,
    # 'custom_surgery_flag':{},
}


#Default Flags for replacement dict with no training required as subset of default flags
# for custom replacement add a custom flag name (any string not in default flag) as key and map it to a dict containing pattern and replacement
# note if same key is used for two pattern the last replacement will be performed
default_replacement_flag_dict_no_training:dict[str,bool|dict] ={
    'relu6_to_relu' : True,
    'dropout_inplace_to_dropout':True,
    'break_maxpool2d_with_kernel_size_greater_than_equalto_5':True,
    'break_avgpool2d_with_kernel_size_greater_than_equalto_5':True,
    'convert_resize_params_size_to_scale':True,
    'promote_conv2d_with_even_kernel_to_larger_odd_kernel':False,
    # 'custom_surgery_flag':{},
}


#Mapping between the flags and the actual replacements corresponding to them
# This dictionary is used whenever a flag is enabled to fetch the corresponding replacement entries
flag_to_dict_entries:dict [str:dict] ={
    'squeeze_and_excite_to_identity' : {SEModule:nn.Identity,SqueezeExcitation:nn.Identity},
    'all_activation_to_relu': {nn.ReLU:nn.ReLU, nn.ReLU6:nn.ReLU, nn.GELU:nn.ReLU, nn.SiLU:nn.ReLU, nn.Hardswish:nn.ReLU, nn.Hardsigmoid:nn.ReLU, nn.LeakyReLU:nn.ReLU,},
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


# returns default dictionary for replacement
def get_replacement_flag_dict_default(return_flags = True, can_retrain = False):
    '''
    returns the default flag dictionary.
    to see the dict print 'default_replacement_flag_dict' from the file this function is in
    '''
    flag_dict = default_replacement_flag_dict_no_training if can_retrain else default_replacement_flag_dict
    repalcement_entries_dict = get_replacement_dict(flag_dict,)
    return flag_dict if return_flags else repalcement_entries_dict


def get_replacement_dict(
    replacement_flag_dict: dict[str|nn.Module|FunctionType|type,bool|nn.Module|FunctionType|type|tuple[FunctionType,FunctionType]]=None,
    can_retrain:bool = True
    ):
    '''
    this function actually converts the flags mapped to True to their corresponding replacements
    if no replacement_flag_dict is given it uses default flag dictionary based on can_retrain
    if the flags is not registered in 'flag_to_dict_entries', its value should be a dict of replacement and that will be updated in the dictionary
    '''
    replacement_flag_dict = replacement_flag_dict or (default_replacement_flag_dict if can_retrain else default_replacement_flag_dict_no_training)
        
    replacement_dict:dict[Any,list[tuple]] = {}
    
    def adjust_value_for_replacement_dict(v1):
        input_adjustment_func = None
        if isinstance(v1,type) and issubclass(v1,nn.Module):
            v1 = v1()
            
        if isinstance(v1,dict):
            assert 'gen_func' in v1 and isinstance(v1['gen_func'],FunctionType), "if value is a dict, it must contain a value for 'gen_func'"
            kwargs = v1.get('kwargs', None)
            v1['func'] = adjust_value_for_replacement_dict(v1['func'])
            input_adjustment_func = v1.get('input_adjustment_func',None) or v1['func'][1]
            v1 = partial(v1['func'][0],**kwargs) if kwargs else v1['func'][0]
        
        if not isinstance(v1,tuple):
            v1 = v1,input_adjustment_func
        else:
            assert len(v1) == 2
            adjusted_v1 =adjust_value_for_replacement_dict(v1[0])
            v1 = adjusted_v1[0],v1[1] or adjusted_v1[1]
        
        return v1
    
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
    keys can be any of a module, a module class, torch.fx wrapped function or a method name of Tensor (str)
    values can be 
    value type                                      perpose
    type                                        ->  converts all partitions of key to the default module of that type
    module                                      ->  converts all partitions of key to a copy of that module
    a function                                  ->  converts all partitions of key to a copy of module generated by the function
                                                    which is traced through input generated from the input nodes to the partition 
    tuple of any of above and a function        ->  converts all partitions of key to a copy of module generated by corresponding method
                                                    which is traced through input generated by second func
    dict(must have a value for 'gen_func',)     ->  converts all partitions of key to a copy of module generated by the gen_Func 
                                                    if 'kwargs' is in dict so the gen_func will have those kwargs
                                                    if 'input_adjustment_func' is in dict, the module will be traced using input generated by it 
                                                    else which is traced through input generated from the input nodes to the partition 
    list of any of above                        ->  converts all partitions of key to a copy of module generated from all the changes in the list 
                                                    if possible in same order
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