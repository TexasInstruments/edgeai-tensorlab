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
from torch import _dynamo as torch_dynamo
from torch import nn , Tensor
import random
from types import FunctionType
import math
from torch.fx import GraphModule
from torch.fx.passes.utils.source_matcher_utils import SourcePartition
from ....xnn.layers import resize_with_scale_factor

# Note: Some of the modules from this file are copied from surgery v2. 
#       So if changes are made in them, those must be done here as well

# focus module for segmentation models
class Focus(nn.Module):
    def forward(self,x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return x
    
    
# focus module for segmentation models
class OptimizedFocus(nn.Module):
    def forward(self,x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = x.permute(0, 3, 1, 2)
        return x


# a typical convulation module to be used as replacement
class ConvBNRModule(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,groups=1) -> None:
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding, groups=groups)
        self.bn=nn.BatchNorm2d(out_channels)
        self.act=nn.ReLU()
    
    def forward(self,x,*args):
        return self.act(self.bn(self.conv(x)))


# custom module for padding
class Padding(nn.Module):
    def __init__(self, pad, mode, value=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pad = pad
        self.mode = mode if mode != 'zeros' else 'constant' 
        self.value = value if mode != 'zeros' else 0.0
    
    def forward(self, x):
        return nn.functional.pad(x,self.pad,self.mode,self.value,)


# custom module for permute
class ReplacementPermute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        
    def forward(self, x):
        return torch.permute(x, self.dims)


# custom module for Upsample with size
class ResizeScaleFactorOnly(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None,
                             recompute_scale_factor=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corner = align_corners
        self.recompute_scale_factor = recompute_scale_factor
    
    def forward(self,x):
        return resize_with_scale_factor(x, self.size, self.scale_factor, self.mode, self.align_corner, self.recompute_scale_factor)


# Wrapper module all replaced module 
class ReplacedModule(nn.Module):
    def __init__(self, main_model:GraphModule, partition:SourcePartition, gen_func:FunctionType, input_adjustment_func:FunctionType, trace_through:bool = True, aten_graph = False, verbose= False, *args, **kwargs) -> None:
        '''
        gen_func: a function that takes main_model and partition as arguments and returns a nn.Module or None
            -> module: that will replace the partition
            -> none: if partition is not required for replacement which can handled later
        '''
        super().__init__(*args, **kwargs)
        self.partition = partition
        self.gen_func = gen_func
        self.input_adjustment_func = input_adjustment_func
        self.inputs = {}
        for node in partition.input_nodes:
            if 'val' in node.meta:
                self.inputs[node] = node.meta['val']
            elif 'example_value' in node.meta:
                self.inputs[node] = node.meta['example_value']
        self.module = self.gen_func(main_model,partition,aten_graph)
        example_args,example_kwargs =self.input_adjustment_func(partition,self.inputs)
        if self.module is not None:
            x = None
            arg_tensors = [x for x in example_args if isinstance(x, torch.Tensor)]
            kwarg_tensors = [x for x in example_kwargs.values() if isinstance(x, torch.Tensor)]
            if (param := next(main_model.parameters(),None)) is not None:
                x = param
            elif len(arg_tensors):   
                x = arg_tensors[0]
            elif len(kwarg_tensors):
                x = kwarg_tensors[0]
            
            if x is not None:
                self.module = self.module.to(device= x.device)
            else:
                print(f'A tensor couldn\'t be found to change the replaced module\'s device from cpu to main model\'s device, the module {partition.source} will not be changed.')
                self.module = None
            
        if trace_through and self.module is not None:
            try:
                self.module, _ = torch_dynamo.export(self.module,aten_graph=aten_graph,assume_static_by_default=True)(*example_args,**example_kwargs)
                num_inputs = len([node for node in self.module.graph.nodes if node.op == 'placeholder'])
                out_nodes = [node for node in self.module.graph.nodes if node.op == 'output']
                num_outputs = len(out_nodes[0].args[0])
                if num_inputs != len(partition.input_nodes) and num_outputs != len(partition.output_nodes):
                    print(f'Replacement has {num_inputs} inputs and {num_outputs} whereas original partition of {partition.source} has {len(partition.input_nodes)} inputs {len(partition.output_nodes)} outputs. So the module {partition.source} will not be changed.')
                    self.module = None
            except Exception as e:
                print(f'While creating a replacement for the partition of {partition.source} found exception {e}. So,the module {partition.source} will not be changed.')
                self.module = None

                

    def __repr__(self):
        self.__class__.__name__ = f'Replaced{self.partition.source.__name__}'
        return super().__repr__()
    
    def forward(self,*args,**kwargs):
        return self.module(*args,**kwargs)
    
