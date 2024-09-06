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
from typing import Union, Dict, Any
from torch.fx import GraphModule
from inspect import isfunction, ismethod

from . import custom_modules, custom_surgery_functions
from .replacer import graph_pattern_replacer,replace_module_nodes,replace_function_nodes
from .custom_symbolic_trace import custom_symbolic_trace

__all__ = ['_replace_unsupported_layers', ]



def _is_replacable(pattern:Union[GraphModule, nn.Module, callable]):
    try:
        if not isinstance(pattern,GraphModule):
            pattern = custom_symbolic_trace(pattern)
    except:
        return False
    return True


def _replace_unsupported_layers(model:nn.Module, example_input:list=[], example_kwargs:dict={}, replacement_dict:Dict[Any,Union[nn.Module,callable]]=None, copy_args:list=[], verbose_mode:bool=False) -> GraphModule | nn.Module:
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
            model = replacement(model, pattern = pattern, example_input = example_input, verbose_mode=verbose_mode)
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
    
    return model





