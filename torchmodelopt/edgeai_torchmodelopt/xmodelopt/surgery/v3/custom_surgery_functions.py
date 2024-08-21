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

from torch import nn
from torch.fx import symbolic_trace, Node
from torch.fx import GraphModule
from torch.fx.passes.utils.source_matcher_utils import SourcePartition
import torch
from copy import deepcopy
import math
import random


from . import replacer
# from .symbolic_trace import symbolic_trace
from . import custom_modules


def gen_func_for_conv2d_kernel_gt_7(main_model:GraphModule, partition:SourcePartition, aten_graph:bool = False,):
    '''
    replacement module generator function, this should generate replacement module 
    for all possible nn.conv2d modules if it is passed to the dictionary
    if kernel size > 7  :   returns a replacement module with appropiate number of ConvBnRelu module until it reaches a kernel size <= 7
                else    :   returns None means no need to replace
    assuming kernel size is same on both axes
    '''
    
    if partition.source != nn.Conv2d:
        return None
    
    nodes = partition.nodes
    if aten_graph:
        params = dict(main_model.named_parameters())
        weight_name = nodes[0].target
        weight = params[weight_name]
        bias_name = nodes[1].target if len(partition.nodes) > 2 else None
        bias = params[bias_name] if bias_name is not None else None
        
        assert len(weight.shape) == 4
        out_channels,in_channels,k_size,_ = list(weight.shape) 
        stride,_ = partition.output_nodes[0].kwargs.get('stride', partition.output_nodes[0].args[3])
        padding = partition.output_nodes[0].kwargs.get('padding',partition.output_nodes[0].args[4])
        padding_mode = partition.output_nodes[0].kwargs.get('padding_mode',partition.output_nodes[0].args[7])
    else:
        modules = dict(main_model.named_modules())
        module = modules[partition.output_nodes[0].target]
        assert isinstance(module, nn.Conv2d)
        weight = module.weight
        bias = module.bias
        in_channels= module.in_channels
        out_channels= module.out_channels
        k_size= module.kernel_size[0]
        stride = module.stride[0]
        padding = module.padding
        padding_mode = module.padding_mode
    
    if padding == 'valid' :
        padding = 0
    elif padding == 'same':
        padding = padding
    else:
        padding = padding[0]
    
    kernel_size = k_size
    
    if k_size <= 7 :
        return None
    replacement = nn.Sequential()
    max_padding = (k_size-1)/2 if k_size %2 == 1 else (k_size/2 -1)  
    
    while k_size>7:
        if  isinstance(padding,str):
            t_padding = padding 
        else:
            if padding > max_padding :
                t_padding = 1
                padding -= 1
            else:
                t_padding = 0
        temp_out_channels= out_channels
        replacement.append(custom_modules.ConvBNRModule(in_channels,temp_out_channels, kernel_size=3,stride=1,padding=t_padding))
        in_channels=temp_out_channels
        k_size-=2
    
    if k_size % 2 == 1:
        padding = padding if isinstance(padding, str) else min(3,padding)
        replacement.append(nn.Conv2d(in_channels,out_channels,k_size,stride=stride,padding=padding))
    else:
        max_padding = min(2, (k_size/2 -1))
        padding =  padding if isinstance(padding, str) else (min(max_padding,padding))
        replacement.append(nn.Conv2d(in_channels,out_channels,kernel_size=k_size,stride=stride,padding=padding))
    
    return replacement

def gen_func_for_conv2d_even_kernel_to_odd(main_model:GraphModule, partition:SourcePartition, aten_graph:bool = False,):
    '''
    replacement module generator function, this should generate replacement module 
    for all possible nn.conv2d modules if it is passed to the dictionary
    if kernel size even :   returns a replacement module with larger odd kernel with same weight
                else    :   returns None means no need to replace
    assuming kernel size is same on both axes
    '''
    
    if partition.source != nn.Conv2d:
        return None
    
    nodes = partition.nodes
    if aten_graph:
        params = dict(main_model.named_parameters())
        weight_name = nodes[0].target
        weight = params[weight_name]
        bias_name = nodes[1].target if len(partition.nodes) > 2 else None
        bias = params[bias_name] if bias_name is not None else None
        
        assert len(weight.shape) == 4
        out_channels,in_channels,k_size,_ = list(weight.shape) 
        stride,_ = partition.output_nodes[0].kwargs.get('stride', partition.output_nodes[0].args[3])
        padding = partition.output_nodes[0].kwargs.get('padding',partition.output_nodes[0].args[4])
        padding_mode = partition.output_nodes[0].kwargs.get('padding_mode',partition.output_nodes[0].args[7])
    else:
        modules = dict(main_model.named_modules())
        module = modules[partition.output_nodes[0].target]
        assert isinstance(module, nn.Conv2d)
        weight = module.weight
        bias = getattr(module,'bias',None) 
        in_channels= module.in_channels
        out_channels= module.out_channels
        k_size= module.kernel_size[0]
        stride = module.stride[0]
        padding = module.padding
        padding_mode = module.padding_mode
    
    if padding == 'valid' :
        padding = 0
    elif padding == 'same':
        padding = padding
    else:
        padding = padding[0]

    if k_size%2 == 1:
        return None
    replacement = nn.Sequential()

    replacement.append(custom_modules.Padding((1,0,1,0),padding_mode,))
    last_conv = nn.Conv2d(in_channels,out_channels,kernel_size=k_size+1,stride=stride,padding=padding)
    
    last_conv.bias = deepcopy(bias) if bias is not None else nn.Parameter(torch.zeros_like(last_conv.bias))
    new_weight = torch.ones_like(last_conv.weight)
    new_weight[:,:,1:,1:] = weight.data
    last_conv.weight = nn.Parameter(new_weight)
    replacement.append(last_conv)
    
    return replacement
    

def gen_func_for_upsample(main_model:GraphModule, partition:SourcePartition, aten_graph:bool = False):
    '''
    replacement module generator function, this should generate replacement module 
    for all possible nn.Upsample modules if it is passed to the dictionary
    if has size factor as  parameter: returns a replacement module with scale factor only
                            else    : returns None means no need to replace
    '''
    if partition.source not in (nn.Upsample,):
        return None
    if aten_graph:
        #TODO make  replacement for upsample
        return None
    else:
        modules = dict(main_model.named_modules())
        module = modules[partition.output_nodes[0].target]
        assert isinstance(module, nn.Upsample)
        size = module.size
        scale_factor = module.scale_factor
        mode = module.mode
        align_corners = module.align_corners
        recompute_scale_factor = module.recompute_scale_factor
    
    if size is None:
        return None
    
    replacement = custom_modules.ResizeScaleFactorOnly(size, scale_factor, mode, align_corners, recompute_scale_factor)
    
    return replacement


def gen_func_for_pool(main_model:GraphModule, partition:SourcePartition, aten_graph:bool = False):
    '''
    replacement module generator function, this should generate replacement module 
    for all possible nn.MaxPool2s or nn.AvgPool2d modules if it is passed to the dictionary
    if kernel size >= 5 :   returns a series of pool with kernel size 3 or 2 
                else    :   returns None means no need to replace
    '''
    if  partition.source not in (nn.MaxPool2d, nn.AvgPool2d) :
        return None
    if aten_graph:
        #TODO make replacement for nn.MaxPool2d, nn.AvgPool2d
        return None
    else:
        modules = dict(main_model.named_modules())
        module = modules[partition.output_nodes[0].target]
        assert isinstance(module, (nn.MaxPool2d, nn.AvgPool2d))
        k_size = module.kernel_size
        stride = module.stride
        padding = module.padding
    
    if k_size < 5 : 
        return None

    replacement = nn.Sequential()
    while k_size > 4 :
        if k_size % 2 == 0:
            replacement.append(partition.source(kernel_size=2, stride=1, padding=(1,1)))
        else: replacement.append(partition.source(kernel_size=3,stride=1,padding=1))
        k_size-=2
    replacement.append(partition.source(kernel_size=k_size, stride=stride, padding=1 if padding % 2 != 0 else (1,1)))

    return replacement

# TODO change the approach for pt2e or not?
def remove_identiy(model:nn.Module, verbose_mode=False, **kwargs):
    model=deepcopy(model)
    traced_model=symbolic_trace(model) if not isinstance(model, torch.fx.GraphModule) else model
    modules= dict(traced_model.named_modules())
    n=0
    nodes=[]
    for node in traced_model.graph.nodes:
        if (node.op == 'call_module') and isinstance(modules[node.target],nn.Identity):
                nodes.append(node)
    for node in nodes:
        try:
            node.replace_all_uses_with(node.args[0])
            copy_found=False
            for node_1 in nodes:
                if node!=node_1 and node.target==node_1.target:
                    copy_found=True
                    break
            if not copy_found:
                parent_name,name=replacer._get_parent_name(node.target)           
                modules[parent_name].__delattr__(name)
                modules.pop(node.target)
            traced_model.graph.erase_node(node)
            n+=1
        except Exception as e:
            if verbose_mode:
                print(n,e)
    traced_model.graph.lint()
    traced_model.recompile()
    if verbose_mode:
        print('Identity removed',n)
    return traced_model


def replace_permute_layer(model:nn.Module, pattern=None, verbose_mode=False):
    model = torch.fx.symbolic_trace(model)
    i = 0
    for node in model.graph.nodes:
        if node.op == 'call_method' and node.target=='permute':
            replacement = custom_modules.ReplacementPermute(node.args[1:])
            prepared_replacement = torch.fx.symbolic_trace(replacement)
            with model.graph.inserting_before(node):
                new_node_name = type(replacement).__name__+str(i)
                model.add_submodule(new_node_name, deepcopy(prepared_replacement))
                new_args = []
                for arg in node.args:
                    if type(arg) == Node:
                        if arg.op != "get_attr":
                            new_args.append(arg)
                        #
                    #
                #
                new_node = model.graph.call_module(new_node_name, tuple(new_args), {})
                node.replace_all_uses_with(new_node)
                model.graph.erase_node(node)
            #
            i+=1    
        #
    #
    model.graph.lint()
    model.recompile()
    if verbose_mode:
        print('reshape/permute : ', i)
        
    return model