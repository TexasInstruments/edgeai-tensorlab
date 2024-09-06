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
from copy import deepcopy
from torch.fx import GraphModule
from torch import _dynamo as torch_dynamo
from types import FunctionType, BuiltinFunctionType
import warnings
from functools import partial


from .utils import get_source_partition

# for repo specific modules 
try:
    from timm.layers.squeeze_excite import SEModule
except:
    # this will make the program skip the search for pattern
    SEModule = None

from . import custom_modules, custom_surgery_functions
from .replacer import graph_pattern_replacer, _remove_hanging_nodes
from . import replacer

# from .custom_symbolic_trace import custom_symbolic_trace

__all__ = ['_replace_unsupported_layers',]


def _replace_unsupported_layers(model:nn.Module, example_input:list=[], example_kwargs:dict={}, replacement_dict:Dict[Any,Union[nn.Module,callable]]=None, aten_graph:bool = False, copy_args:list=[], verbose_mode:bool=False):
    
    
    # assuming if it is a graph module it is generated through dynamo export 
    # TODO make symbolic trace generated module is goes through dynamo export
    traced_model,_ =(model,None) if isinstance(model,GraphModule) else torch_dynamo.export(model,aten_graph=aten_graph,assume_static_by_default=True)(*example_input,**example_kwargs) 
    
    replacer.__net_module_replaced = 0
    
    for pattern, replacement in replacement_dict.items():
        if pattern is None:
            continue
        
        # class of Module of
        if isinstance(pattern, nn.Module):
            pattern = type(pattern)
        source_partiions = get_source_partition(traced_model.graph,[pattern])

        if pattern not in source_partiions:
            continue
        
        # calls the main surgery function
        traced_model = graph_pattern_replacer(traced_model, source_partiions[pattern], replacement,aten_graph= aten_graph, verbose_mode=verbose_mode)
        traced_model(*example_input,**example_kwargs)
        # print(traced_model.graph)
        
    _remove_hanging_nodes(traced_model) 
    
    return traced_model




