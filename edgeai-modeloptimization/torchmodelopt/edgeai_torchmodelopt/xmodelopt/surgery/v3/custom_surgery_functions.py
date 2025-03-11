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


def gen_func_for_conv2d_kernel_gt_7(main_model:GraphModule, partition:SourcePartition, aten_graph:bool = True,):
    '''
    replacement module generator function, this should generate replacement module 
    for all possible nn.conv2d modules if it is passed to the dictionary
    if kernel size > 7  :   returns a replacement module with appropiate number of ConvBnRelu module until it reaches a kernel size <= 7
                else    :   returns None means no need to replace
    assuming kernel size is same on both axes
    '''
    
    if partition.source != nn.Conv2d:
        return None
    
    conv_node = partition.output_nodes[0]
    if aten_graph:
        params = dict(main_model.named_parameters())
        weight = params[conv_node.args[1].target]
        num_args = len(conv_node.args)
        assert len(weight.shape) == 4
        out_channels, in_channels, *k_size = list(weight.shape) 
        bias_node = conv_node.args[2] if num_args >= 3 else None
        bias = params[bias_node.target] if bias_node is not None else None
        stride = conv_node.args[3] if num_args >= 4 else [1, 1]
        padding = conv_node.args[4] if num_args >= 5 else [0, 0]

    else:
        modules = dict(main_model.named_modules())
        module = modules[conv_node.target]
        assert isinstance(module, nn.Conv2d)
        weight = module.weight
        bias = module.bias
        in_channels = module.in_channels
        out_channels = module.out_channels
        k_size = module.kernel_size    
        stride = module.stride
        padding = module.padding
        padding_mode = module.padding_mode
    
    if isinstance(k_size, (tuple, list)):
        assert len(k_size) == 2 and k_size[0] == k_size[1]
        k_size = k_size[0]
    
    if isinstance(stride, (tuple, list)):
        assert len(stride) == 2 and stride[0] == stride[1]
        stride = stride[0]
    
    if isinstance(padding, (tuple, list)):
        assert len(padding) == 2 and padding[0] == padding[1]
        padding = padding[0]
    
    if k_size <= 7:
        return None
    replacement = nn.Sequential()
    max_padding = (k_size - 1) / 2 if k_size % 2 == 1 else (k_size / 2 - 1)  
    
    while k_size > 7:
        if isinstance(padding, str):
            t_padding = padding 
        else:
            if padding > max_padding:
                t_padding = 1
                padding -= 1
            else:
                t_padding = 0
        temp_out_channels = out_channels
        replacement.append(custom_modules.ConvBNRModule(in_channels, temp_out_channels, kernel_size=3, stride=1, padding=t_padding))
        in_channels = temp_out_channels
        k_size -= 2
    
    if k_size % 2 == 1:
        padding = padding if isinstance(padding, str) else min(3, padding)
        replacement.append(nn.Conv2d(in_channels, out_channels, k_size, stride=stride, padding=padding))
    else:
        max_padding = min(2, (k_size / 2 - 1))
        padding = padding if isinstance(padding, str) else min(max_padding, padding)
        replacement.append(nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=padding))
    
    return replacement

def gen_func_for_conv2d_even_kernel_to_odd(main_model:GraphModule, partition:SourcePartition, aten_graph:bool = True,):
    '''
    replacement module generator function, this should generate replacement module 
    for all possible nn.conv2d modules if it is passed to the dictionary
    if kernel size even :   returns a replacement module with larger odd kernel with same weight
                else    :   returns None means no need to replace
    assuming kernel size is same on both axes
    '''
    
    if partition.source != nn.Conv2d:
        return None
    
    conv_node = partition.output_nodes[0]
    if aten_graph:
        params = dict(main_model.named_parameters())
        weight = params[conv_node.args[1].target]
        num_args = len(conv_node.args)
        assert len(weight.shape) == 4
        out_channels, in_channels, *k_size = list(weight.shape) 
        if num_args >= 3:
            bias_node = conv_node.args[2]
            bias = params[bias_node.target] if bias_node is not None else None
        stride = conv_node.args[3] if num_args >= 4 else [1, 1]
        padding = conv_node.args[4] if num_args >= 5 else [0, 0]

    else:
        modules = dict(main_model.named_modules())
        module = modules[conv_node.target]
        assert isinstance(module, nn.Conv2d)
        weight = module.weight
        bias = module.bias
        in_channels = module.in_channels
        out_channels = module.out_channels
        k_size = module.kernel_size    
        stride = module.stride
        padding = module.padding
        padding_mode = module.padding_mode
    
    if isinstance(k_size,(tuple, list)):
        assert len(k_size) == 2 and k_size[0] == k_size[1]
        k_size = k_size[0]
    
    if isinstance(stride,(tuple, list)):
        assert len(stride) == 2 and stride[0] == stride[1]
        stride = stride[0]
    
    if isinstance(padding,(tuple, list)):
        assert len(padding) == 2 and padding[0] == padding[1]
        padding = padding[0]

    if k_size % 2 == 1:
        return None
    replacement = nn.Sequential()

    replacement.append(custom_modules.Padding((1, 0, 1, 0), padding_mode))
    last_conv = nn.Conv2d(in_channels, out_channels, kernel_size=k_size + 1, stride=stride, padding=padding)
    
    last_conv.bias = deepcopy(bias) if bias is not None else nn.Parameter(torch.zeros_like(last_conv.bias))
    new_weight = torch.ones_like(last_conv.weight)
    new_weight[:, :, 1:, 1:] = weight.data
    last_conv.weight = nn.Parameter(new_weight)
    replacement.append(last_conv)
    
    return replacement
    

def gen_func_for_upsample(main_model:GraphModule, partition:SourcePartition, aten_graph:bool = True):
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
        mode_to_func_map = {
            'nearest': (torch.ops.aten.upsample_nearest1d.vec, torch.ops.aten.upsample_nearest2d.vec,torch.ops.aten.upsample_nearest3d.vec,),
            'linear': (torch.ops.aten.upsample_linear1d.vec,),
            'linear': (torch.ops.aten.upsample_bilinear2d.vec,),
            'linear': (torch.ops.aten.upsample_trilinear3d.vec,),
            'linear': (torch.ops.aten.upsample_bicubic2d.vec,),
        }
        upsample_node = partition.output_nodes[0]
        mode = None
        for k, v in mode_to_func_map.items():
            if upsample_node.target in v:
                mode = k
        if mode is None:
            return None
        if mode == 'nearest':
            size= upsample_node.args[1]
            align_corners = False
            scale_factor= upsample_node.args[2]
        else:
            size= upsample_node.args[1]
            align_corners = upsample_node.args[2]
            scale_factor= upsample_node.args[3]
    
    else:
        modules = dict(main_model.named_modules())
        module = modules[partition.output_nodes[0].target]
        assert isinstance(module, nn.Upsample)
        size = module.size
        scale_factor = module.scale_factor
        mode = module.mode
        align_corners = module.align_corners
        recompute_scale_factor = module.recompute_scale_factor 
    
    recompute_scale_factor = False # if left True causes Upsample with size not scale
    
    if size is None:
        return None
    
    replacement = custom_modules.ResizeScaleFactorOnly(size, scale_factor, mode, align_corners, recompute_scale_factor)
    
    return replacement


def gen_func_for_pool(main_model:GraphModule, partition:SourcePartition, aten_graph:bool = True):
    '''
    replacement module generator function, this should generate replacement module 
    for all possible nn.MaxPool2s or nn.AvgPool2d modules if it is passed to the dictionary
    if kernel size >= 5 :   returns a series of pool with kernel size 3 or 2 
                else    :   returns None means no need to replace
    '''
    if partition.source not in (nn.MaxPool2d, nn.AvgPool2d):
        return None
    
    pool_node = partition.output_nodes[0]
    
    if aten_graph:
        num_args = len(pool_node.args)
        kernel_size = pool_node.args[1]
        stride = pool_node.args[2] if num_args >= 3 else kernel_size
        padding = pool_node.args[3] if num_args >= 4 else 0
    else:
        modules = dict(main_model.named_modules())
        module = modules[pool_node.target]
        assert isinstance(module, (nn.MaxPool2d, nn.AvgPool2d))
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding
    
    if isinstance(kernel_size, (tuple,list)) and len(kernel_size) == 2:
        assert kernel_size[0] == kernel_size[1], "In Pooling, kernel_size must be a square" 
        k_size = kernel_size[0]
    else:
        k_size = kernel_size        
    
    if isinstance(stride, (tuple,list)) and len(stride) == 2:
        assert stride[0] == stride[1], "In Pooling, kernel_size must be a square" 
        stride = stride[0]
    else:
        stride = stride  
    
    if isinstance(padding, (tuple,list)) and len(padding) == 2:
        assert padding[0] == padding[1], "In Pooling, kernel_size must be a square" 
        padding = padding[0]
    else:
        padding = padding    
    
    if k_size < 5: 
        return None

    replacement = nn.Sequential()
    
    while k_size > 4:
        if k_size % 2 == 0:
            replacement.append(partition.source(kernel_size=2, stride=1, padding=(1, 1)))
        else:
            replacement.append(partition.source(kernel_size=3, stride=1, padding=1))
        k_size -= 2
    
    replacement.append(partition.source(kernel_size=k_size, stride=stride, padding=1 if padding % 2 != 0 else (1, 1)))

    return replacement

