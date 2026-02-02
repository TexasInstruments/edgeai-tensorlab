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

import operator
from typing import  Type, List, Dict, Any, Iterable
import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.passes.utils.source_matcher_utils import  SourcePartition

filter_funcs_dict = dict()

def register_filter(*args, func=None):
    def _registered(func):
        filter_funcs_dict[args] = func
        return func
    if func  is not None:
        return _registered(func)
    return _registered

def register_n2m_filter(key,  n, m, func=None):
    return register_filter(key,  n, m, 'n2m', func=func)

def register_n2m_filters(n, m):
    #Note: Consider these functions as examples for further filter functions as they need to return nodes for next steps
    @register_n2m_filter('Conv2d', n, m)
    def convs_filter_func(module):
        assert isinstance(module, (fx.GraphModule)), f'GraphModule object should be given! but got object of type {module.__class__.__name__}'
        ret = []
        params = dict(module.named_parameters())
        graph = module.graph 
        for node in graph.nodes:
            if node.target != torch.ops.aten.conv2d.default:
                continue
            weight = node.args[1]
            if weight.op != 'get_attr':
                # Skipping as weight tensor is variable to conv
                continue
            weight = params.get(weight.target)
            out_channel, in_channel, *kernel_size = weight.shape
            if out_channel % m != 0 or in_channel % m != 0 :#or not all(k==1 for k in kernel_size ):
                # skipping as conv is not supported for n:m sparsity
                continue
            ret.append([node])
        
        return ret

    @register_n2m_filter('Linear', n, m)
    def linears_filter_func(module):
        assert isinstance(module, (fx.GraphModule)), f'GraphModule object should be given! but got object of type {module.__class__.__name__}'
        ret = []
        params = dict(module.named_parameters())
        graph = module.graph 
        for node in graph.nodes:
            if node.target != torch.ops.aten.linear.default:
                continue
            weight = node.args[1]
            if weight.op != 'get_attr':
                # Skipping as weight tensor is variable to conv
                continue
            weight = params.get(weight.target)
            out_channel, in_channel= weight.shape
            if out_channel % m != 0 or in_channel % m != 0 :
                # skipping as linear is not supported for n:m sparsity
                continue
            ret.append([node])
        
        return ret

    @register_n2m_filter('matmul', n, m)
    def linears_filter_func(module):
        assert isinstance(module, (fx.GraphModule)), f'GraphModule object should be given! but got object of type {module.__class__.__name__}'
        ret = []
        params = dict(module.named_parameters())
        graph = module.graph 
        for node in graph.nodes:
            if node.target != torch.ops.aten.matmul.default:
                continue
            
            weight = node.args[1]
            if weight.op == 'get_attr':
                weight = params.get(weight.target)
                out_channel, in_channel= weight.shape
                if out_channel % m == 0 and in_channel % m == 0 :
                    ret.append([node])
            weight = node.args[0]
            if weight.op == 'get_attr':
                weight = params.get(weight.target)
                out_channel, in_channel= weight.shape
                if out_channel % m == 0 and in_channel % m == 0 :
                    ret.append([node]) if [node] not in ret else None
            
        
        return ret


def get_sparsity_nodes(module: fx.GraphModule, *args ):
    ret = {}
    for key,func in filter_funcs_dict.items():
        if args and not all(a in key for a in args):
            continue
        ret[key] = func(module)
    return ret

weight_func_dict = {}

def register_weigth_func(*args, func=None):# -> Any | Callable[..., Any]:
    def _registered(func):
        weight_func_dict[args] = func
        return func
    if func  is not None:
        return _registered(func)
    return _registered

def register_n2m_weight_func(key, n, m, func=None):
    return register_weigth_func(key, n, m, 'n2m', func=func)

def register_n2m_weight_funcs(n,m):
    #Note: Consider these functions as examples for further weight functions 
    
    # either a list/tuple or set is expected if the weights are used to mask only themselves
    # otherwise  a dictionary mapping the weights to create mask to list of weights the mask will be applied 
    # (they themselves have to be included)
    # currently only a single weight can be used to create mask
    # TODO if required, implement the flow for multiple weights to impact mask generation
    
    # similarly, currently mask can only be multiplied with the applied weight directly in parametrization
    # TODO if required, implement the flow for different type of weights to be multiplied to mask
    @register_n2m_weight_func('Conv2d', n, m)
    def get_conv_weights( module: fx.GraphModule, nodes:list[fx.Node]):
        conv = nodes[0]
        weight = conv.args[1]
        return [weight.target]
    
    @register_n2m_weight_func('Linear', n, m)
    def get_linear_weights( module: fx.GraphModule, nodes:list[fx.Node]):
        fc = nodes[0]
        weight = fc.args[1]
        return [weight.target]
    
    @register_n2m_weight_func('matmul', n, m)
    def get_linear_weights( module: fx.GraphModule, nodes:list[fx.Node]):
        matmul = nodes[0]
        return [a.target for a in matmul.args if a.op == 'get_attr']
    

def get_all_weights(module: fx.GraphModule, nodes_dict:dict[tuple,list[fx.Node]],):
    ret = {}
    for key, nodes_list in nodes_dict.items():
        results = []
        
        for nodes in nodes_list:
            results.append(weight_func_dict[key](module, nodes, ))
        
        ret[key]= results
    return ret