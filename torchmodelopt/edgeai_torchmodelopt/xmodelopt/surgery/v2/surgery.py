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
from torchvision import ops
from typing import Union, Dict, Any
from copy import deepcopy
from torch.fx import GraphModule, symbolic_trace
from inspect import isfunction, ismethod

try:
    from timm.layers.squeeze_excite import SEModule
except:
    SEModule = None

from . import custom_modules, custom_surgery_functions
from .replacer import graph_pattern_replacer,replace_module_nodes,replace_function_nodes
from .custom_symbolic_trace import custom_symbolic_trace

__all__ = ['replace_unsupported_layers', 'get_replacement_dict_default','SurgeryModule']


#put composite modules first, then primary module
_unsupported_module_dict={
    SEModule:nn.Identity,                                       # timm specific
    custom_modules.SEModule() : nn.Identity(),
    custom_modules.SEModule1() : nn.Identity(),
    # SEModule(32) : nn.Identity(),                             # timm specific
    # SEModule(32,gate_layer=nn.Hardsigmoid) : nn.Identity(),   # timm specific
    # SEModule(32,act_layer=nn.SiLU) : nn.Identity(),           # timm specific
    'CNBlock':custom_surgery_functions.replace_cnblock,         # for convnext of torch vision
    # 'SELzyer':custom_surgery_functions.replace_se_layer,
    nn.ReLU(inplace=True):nn.ReLU(),
    nn.Hardswish():nn.ReLU(),
    nn.ReLU6():nn.ReLU(),
    nn.ReLU():nn.ReLU(), # check if it might be required for QAT, if ranges are different #TODO
    # nn.GELU():nn.ReLU(),
    nn.SiLU():nn.ReLU(),
    nn.Hardsigmoid():nn.ReLU(),
    nn.LeakyReLU():nn.ReLU(),
    nn.Dropout(inplace=True):nn.Dropout(),
    custom_modules.Focus():custom_modules.ConvBNRModule(3,12,(5,5),(2,2),2), # will only effective if focus appears jus after the input
    # 'layerNorm':custom_surgery_functions.replace_layer_norm, # not effective if len(input.shape) != 4 till date
    'upsample':custom_surgery_functions.replace_resize_with_scale_factor, # for segmentation model -> deeplabv3
    'maxpool_ge_5':custom_surgery_functions.replace_maxpool2d_kernel_size_ge_5, # for segmentation model -> deeplabv3
    'avgpool_ge_5':custom_surgery_functions.replace_avgpool2d_kernel_size_ge_5,
    'conv_ge_7':custom_surgery_functions.replace_conv2d_kernel_size_gt_7,
    'conv_6':custom_surgery_functions.replace_conv2d_kernel_size_6,
}


def _is_replacable(pattern:Union[GraphModule, nn.Module, callable]):
    try:
        if not isinstance(pattern,GraphModule):
            pattern = custom_symbolic_trace(pattern)
    except:
        return False
    return True


def replace_unsupported_layers(model:nn.Module, replacement_dict:Dict[Any,Union[nn.Module,callable]]=None, copy_args:list=[], example_input=None, verbose_mode:bool=False):
    '''
    main function that does the surgery

    it does default surgery if no replacement dictionry is given
    replacement dictionry may contain
    keys                value
    callable        ->  callable            : any call function to call_function if they take same argument partial agument may -                                         be used 
    callable        ->  nn.Module           : any call function to call_function if they take same argument partial agument may -                                         be used 
    Any             ->  Callable            : any self-made surgery function 
    nn.Module       ->  nn.Module           : any nn.Module pattern to replace with another nn.Module
    type            ->  type/nn.Module      : replaces sub-module of same type as patttern using traditional python approach 
    '''
    
    if model.training:
        RuntimeWarning("The model is in train mode, converting to eval mode. This might change the network behaviour.")
        model.eval()
        is_train_mode = True
    else:
        is_train_mode = False
        
    replacement_dict = replacement_dict or _unsupported_module_dict
    model = deepcopy(model)

    for pattern, replacement in replacement_dict.items():
        if pattern is None:
            continue

        if isfunction(pattern) or type(pattern).__name__ in ('builtin_function_or_method','function'):
            # replacement must be partially defined function or work with same args and kwargs
            if isinstance(replacement, (list, tuple)):
                kwargs = replacement[1] if len(replacement) > 1 else None
                replacement = replacement[0]
            else:
                kwargs = dict()
            model = replace_function_nodes(model, pattern, replacement, verbose_mode=verbose_mode, **kwargs)
        elif isfunction(replacement) or ismethod(replacement):
            # for self-made surgery function 
            model = replacement(model, pattern = pattern, example_input = example_input,verbose_mode=verbose_mode)
        else:
            # class of MOdule of
            if isinstance(pattern, type):
                replace_module_nodes(model, pattern, replacement, copy_args=copy_args, verbose_mode=verbose_mode)
            else:
                # for nn.Module
                if pattern.__class__.__name__ in dir(torch.nn):
                    # if the pattern is present in nn directory,
                    # a wrapper module is required, for successful 
                    # surgery on that module
                    model = graph_pattern_replacer(model, pattern, replacement, verbose_mode=verbose_mode)
                    pattern = custom_modules.InstaModule(pattern)

                # calls the main surgery function
                model = graph_pattern_replacer(model, pattern, replacement, verbose_mode=verbose_mode)
    model = custom_surgery_functions.remove_identiy(model)
    
    if is_train_mode:
        model.train()
        
    return model


# returns default dictionary for replacement
def get_replacement_dict_default():
    return _unsupported_module_dict


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

