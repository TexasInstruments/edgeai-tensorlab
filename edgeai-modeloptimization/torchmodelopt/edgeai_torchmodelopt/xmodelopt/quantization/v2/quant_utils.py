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

import os
import copy
import torch
from torch.ao.quantization import quantize_fx
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization import FakeQuantize
import statistics
from functools import partial
from torch.onnx import symbolic_helper
from torch.onnx._internal import jit_utils
from torch import nn

from .... import xnn

from . import observer_types
from . import fake_quantize_types
from . import qconfig_types
import warnings


def is_fake_quant_with_param(self, pmodule, cmodule, fake_quant_types):
    num_params = len(list(pmodule.parameters(recurse=False)))
    return isinstance(cmodule, fake_quant_types) and num_params > 0


def quantized_softmax(g: jit_utils.GraphContext, x, dim, op_scale, op_zero_point):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    output = g.op("Softmax", x)
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


def quantized_matmul(g: jit_utils.GraphContext, x, y, op_scale, op_zero_point):
    x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    y, _, _, _ = symbolic_helper.dequantize_helper(g, y)
    output = g.op("MatMul", x, y)
    return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)

def aten_quantize_channel(g, x, op_scale, op_zero_point, axis, dtype, *args):
    # Tensor input, Tensor scales, Tensor zero_points, int axis, int quant_min, int quant_max, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)
    # x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
    # return x
    
    dtype = symbolic_helper._get_const(dtype, "i", "dtype")
    op_zero_point = g.op("Cast", op_zero_point, to_i=symbolic_helper.scalar_type_to_onnx[dtype])
    op_scale = g.op("Cast", op_scale, to_i=torch.onnx.TensorProtoDataType.FLOAT)
    
    return symbolic_helper.quantize_helper(g, x, op_scale, op_zero_point, axis)
    

@torch.fx.wrap
def _get_rel_pos_bias(relative_position_bias_table, relative_position_index, window_area) -> torch.Tensor:
    relative_position_bias = relative_position_bias_table[
        relative_position_index.view(-1)].view(window_area, window_area, -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = torch.permute(relative_position_bias, (2, 0, 1))  # nH, Wh*Ww, Wh*Ww
    relative_position_bias = relative_position_bias.reshape(1, -1, window_area, window_area)
    return relative_position_bias
    
    
# similar class as to default attention, however it supports model export as well as quantizing matmul operation
class QuantAttention(nn.Module):
    
    def __init__(
            self,
            dim: int = 128,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.relative_position_bias_table = None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, N, C = x.shape
        qkv = torch.permute(self.qkv(x).reshape(B, N, 3, self.num_heads, -1), (2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        #q = torch.mul(q, torch.tensor(self.scale))
        q = torch.mul(q, torch.tensor([self.scale]*self.head_dim))
        attn = torch.matmul(q, k.transpose(-2, -1))
        if self.relative_position_bias_table is not None:
            attn = attn + _get_rel_pos_bias(self.relative_position_bias_table, self.relative_position_index, self.window_area)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = torch.matmul(attn, v)

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


@torch.fx.wrap
def LayerNormWithoutGB(x, eps):
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.pow(x - mean, 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps)


class QuantLayerNorm(torch.nn.Module):
    
    def __init__(self, 
                 normalized_shape = tuple(), 
                 eps: float = 1e-5, 
                 elementwise_affine: bool = True,
                 device=None, dtype=None):
        super().__init__()
                
    def forward(self, x):
        y = LayerNormWithoutGB(x, self.eps)
        y = torch.mul(y, self.weight)
        y = torch.add(y, self.bias)
        return y


def is_mlp_fc2_layer(all_modules, node, find_level, found_gelu=False):
    if find_level<0:
        return False
    elif node.target in all_modules:
        if isinstance(all_modules[node.target], nn.Linear):
            if found_gelu: 
                return True
            else: 
                # found linear before the gelu layer
                return False
            #
        #
        elif isinstance(all_modules[node.target], nn.GELU):
            found_gelu = True
        #
    #
    return is_mlp_fc2_layer(all_modules, node.args[0], find_level-1, found_gelu)


def _bias_calibration_hook(m, x, y, calibration_factor, bias_module):
    bias_error = 0
    if isinstance(x, tuple):
        x = x[0]
    if len(x.shape) == 3:
        float_mean = x.mean(dim=(0,1))
        quant_mean = y.mean(dim=(0,1))
        bias_error = float_mean - quant_mean 
    elif len(x.shape) == 4:
        if x.shape[1] == bias_module.bias.shape[0]:
            float_mean = x.mean(dim=(0,2,3))
            quant_mean = y.mean(dim=(0,2,3)) 
            bias_error = float_mean - quant_mean                                          
        elif x.shape[3] == bias_module.bias.shape[0]:
            float_mean = x.mean(dim=(0,1,2))    
            quant_mean = y.mean(dim=(0,1,2))  
            bias_error = float_mean - quant_mean 
    bias_module.bias.data += (bias_error * calibration_factor)
    return y


def add_bias_calibration_hook(model, calibration_factor=0):
    all_modules = dict(model.named_modules())
    all_hooks = []
    for node in model.graph.nodes:
        if (node.prev.target in all_modules) and (node.target in all_modules):
            bias_module = all_modules[node.prev.target]
            if getattr(bias_module, 'bias', None) is None and hasattr(bias_module, 'bn'):
                bias_module = bias_module.bn
            if getattr(bias_module, 'bias', None) is not None:
                fake_quantize_module = all_modules[node.target]
                _bias_calibration_hook_binded = partial(_bias_calibration_hook, \
                    calibration_factor=calibration_factor, bias_module=bias_module)
                this_hook = fake_quantize_module.register_forward_hook(_bias_calibration_hook_binded)
                all_hooks.append(this_hook)
    return all_hooks
