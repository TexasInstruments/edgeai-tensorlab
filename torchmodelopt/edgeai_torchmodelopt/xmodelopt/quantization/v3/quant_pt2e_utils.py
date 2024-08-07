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
import statistics
from torch.onnx import symbolic_helper, register_custom_op_symbolic, _type_utils
from torch.onnx._internal import jit_utils
from torch import nn
from torch import fx
from functools import partial
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions
import itertools
from torch.fx import Node

from . import fake_quantize_types
from . import qconfig_types
import warnings

def adjust_gradual_quantization(self):
    warnings.warn("Adjust Gradual Quantization is not implemented")
    return
    
    '''
    adjust quantization parameters on epoch basis
    '''
    if self.__quant_params__.qconfig_mode != qconfig_types.QConfigMode.DEFAULT and self.__quant_params__.total_epochs >= 10:
        # find unstable layers and freeze them
        self.adaptive_freeze_layers(fake_quantize_types.ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES)


def is_fake_quant_with_param(self, pmodule, cmodule, fake_quant_types):
    num_params = len(list(pmodule.parameters(recurse=False)))
    return isinstance(cmodule, fake_quant_types) and num_params > 0


def adaptive_freeze_layers(self, fake_quant_types, **kwargs):
    epoch_gradual_quant_start = max(self.__quant_params__.total_epochs // 2, 1)
    if self.__quant_params__.qconfig_mode == qconfig_types.QConfigMode.FREEZE_DEPTHWISE_LAYERS:
        num_total_layers = 0
        self.__quant_params__.forzen_layer_names_list = []
        is_freezing_epoch = (self.__quant_params__.num_epochs_tracked >= epoch_gradual_quant_start)
        for pname, pmodule in list(self.named_modules()):
            is_input_conv_module = False
            is_depthwise_conv_module = False
            if isinstance(pmodule, torch.nn.Conv2d) and pmodule.in_channels < 8:
                # too less input channels, could be first conv module
                is_input_conv_module = True
            if isinstance(pmodule, torch.nn.Conv2d) and pmodule.groups == pmodule.in_channels:
                is_depthwise_conv_module = True
            #
            for cname, cmodule in list(pmodule.named_children()):
                if self.__quant_params__.is_fake_quant_with_param(pmodule, cmodule, fake_quant_types):
                    is_frozen_layer = (is_input_conv_module or is_depthwise_conv_module)
                    if is_freezing_epoch and is_frozen_layer:
                        # stop updating quantization ranges and stats
                        pmodule.apply(torch.ao.quantization.disable_observer)
                        pmodule.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                        # stop updating parmeters
                        for param in pmodule.parameters(recurse=False):
                            param.requires_update = False
                        #
                        self.__quant_params__.forzen_layer_names_list.append(pname)
                    #
                    num_total_layers += 1
                #
            #
        #
        print(f"using adaptive quantization - qconfig_mode:{self.__quant_params__.qconfig_mode} "
              f"num_frozen_layers:{len(self.__quant_params__.forzen_layer_names_list)}/{num_total_layers} ")
    elif self.__quant_params__.qconfig_mode == qconfig_types.QConfigMode.FREEZE_UNSTABLE_LAYERS:
        num_total_layers = 0
        delta_change_list = []
        for pname, pmodule in list(self.named_modules()):
            for cname, cmodule in list(pmodule.named_children()):
                if is_fake_quant_with_param(pmodule, cmodule, fake_quant_types):
                    cmodule.set_adaptive_params(detect_change=True, **kwargs)
                    delta_change_list.append(cmodule.delta_change)
                #
            #
        #
        if self.__quant_params__.num_epochs_tracked >= epoch_gradual_quant_start:
            is_freezing_start_epoch = (self.__quant_params__.num_epochs_tracked == epoch_gradual_quant_start)
            # find sign_change_threshold
            freeze_fraction = 0.15
            delta_change_min = 0.04
            topk_index = int((len(delta_change_list) - 1) * (1 - freeze_fraction))
            delta_change_knee = sorted(delta_change_list)[topk_index]
            delta_change_threshold = max(delta_change_knee, delta_change_min)

            # freeze layers with high sign change
            num_total_layers = 0
            for pname, pmodule in list(self.named_modules()):
                max_delta_change = 0.0
                for cname, cmodule in list(pmodule.named_children()):
                    if is_fake_quant_with_param(pmodule, cmodule, fake_quant_types):
                        # once frozen, always frozen
                        is_frozen_layer = (pname in self.__quant_params__.forzen_layer_names_list)
                        is_high_change = is_freezing_start_epoch and (cmodule.delta_change >= delta_change_threshold)
                        if is_frozen_layer or is_high_change:
                            # stop updating delta_change
                            cmodule.set_adaptive_params(detect_change=False, **kwargs)
                            # stop updating quantization ranges and stats
                            pmodule.apply(torch.ao.quantization.disable_observer)
                            pmodule.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                            # stop updating parmeters
                            for param in pmodule.parameters(recurse=False):
                                param.requires_update = False
                            #
                            self.__quant_params__.forzen_layer_names_list.append(pname)
                        #
                        num_total_layers += 1
                    #
                #
            #
        #
        self.__quant_params__.forzen_layer_names_list = list(set(self.__quant_params__.forzen_layer_names_list))
        print(f"using adaptive quantization - qconfig_mode:{self.__quant_params__.qconfig_mode} "
              f"median_delta_change:{statistics.median(delta_change_list):.4f} max_delta_change:{max(delta_change_list):.4f} "
              f"num_frozen_layers:{len(self.__quant_params__.forzen_layer_names_list)}/{num_total_layers} "
              f"frozen_layers:{self.__quant_params__.forzen_layer_names_list} ")
    #


# def quantized_softmax(g: jit_utils.GraphContext, x, dim, op_scale, op_zero_point):
#     x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
#     output = g.op("Softmax", x)
#     return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


# def quantized_matmul(g: jit_utils.GraphContext, x, y, op_scale, op_zero_point):
#     x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
#     y, _, _, _ = symbolic_helper.dequantize_helper(g, y)
#     output = g.op("MatMul", x, y)
#     return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)


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

        q = torch.mul(q, torch.tensor(self.scale))
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
    var = torch.square(x - mean).mean(dim=-1, keepdim=True)
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
    
    
def register_onnx_symbolics():
    
    def aten_softmax(g: jit_utils.GraphContext, input, dim, *args):
        output = g.op("Softmax", input) #FIXME need to pass dim as well, TIDL needs axis 
        return output

    def aten_unsafe_view(g, x, dim, *args):
        output = g.op("Reshape", x, dim)
        return output
        
    def quantized_decomposed_quantize(g, x, op_scale, op_zero_point, quant_min, quant_max, dtype, *args):
        # Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None
        # x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
        # return x
        dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        op_zero_point = g.op("Cast", op_zero_point, to_i=symbolic_helper.scalar_type_to_onnx[dtype])
        op_scale = g.op("Cast", op_scale, to_i=torch.onnx.TensorProtoDataType.FLOAT)
        
        return symbolic_helper.quantize_helper(g, x, op_scale, op_zero_point)

    def quantized_decomposed_dequantize(g, x, op_scale, op_zero_point, *args):
        # Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None
        x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
        return x

    def quantized_decomposed_quantize_channel(g, x, op_scale, op_zero_point, axis, quant_min, quant_max, dtype, *args):
        # Tensor input, Tensor scales, Tensor zero_points, int axis, int quant_min, int quant_max, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)
        # x, _, _, _ = symbolic_helper.dequantize_helper(g, x)
        # return x
        
        dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        op_zero_point = g.op("Cast", op_zero_point, to_i=symbolic_helper.scalar_type_to_onnx[dtype])
        op_scale = g.op("Cast", op_scale, to_i=torch.onnx.TensorProtoDataType.FLOAT)
        
        return symbolic_helper.quantize_helper(g, x, op_scale, op_zero_point, axis)
    
    def aten_copy(g, x):
        return x
    
    
    register_custom_op_symbolic(
        symbolic_name='aten::lift_fresh_copy',
        symbolic_fn=aten_copy,
        opset_version=17
    )

    register_custom_op_symbolic(
        symbolic_name='aten::_softmax', 
        symbolic_fn=aten_softmax, 
        opset_version=17)
    
    register_custom_op_symbolic(
        symbolic_name='aten::_unsafe_view', 
        symbolic_fn=aten_unsafe_view, 
        opset_version=17)

    register_custom_op_symbolic(
        symbolic_name='quantized_decomposed::quantize_per_tensor', 
        symbolic_fn=quantized_decomposed_quantize, 
        opset_version=17)

    register_custom_op_symbolic(
        symbolic_name='quantized_decomposed::dequantize_per_tensor', 
        symbolic_fn=quantized_decomposed_dequantize, 
        opset_version=17)

    register_custom_op_symbolic(
        symbolic_name='quantized_decomposed::quantize_per_channel', 
        symbolic_fn=quantized_decomposed_quantize_channel, 
        opset_version=17)

    register_custom_op_symbolic(
        symbolic_name='quantized_decomposed::dequantize_per_channel', 
        symbolic_fn=quantized_decomposed_dequantize, 
        opset_version=17)
    
    
def remove_loss_branch(model): 
    # loss branch exists in the model definition, as well as we are supporting it for model training, however, 
    # the branch needs to be removed for onnx export, replacing the branch with identity
    for node in model.graph.nodes:
        if node.target=='output' and len(node.args[0])>1:
            # output node has more than one input branches
            assert ('dequantize' in node.args[0][0].name) or ('dequantize' in node.args[0][1].name), \
                print("dequantize does not exist in the output branch, there could be some error") 
            fc_out_node = node.args[0][0] if ('dequantize' in node.args[0][0].name) else node.args[0][1]
            loss_end_node = node.args[0][0] if ('dequantize' not in node.args[0][0].name) else node.args[0][1]
            # assumption that the output of the network(logits) would be quantized
            loss_node = next((user for user in fc_out_node.users if user.name!='output'), None)
            if loss_node is None: # the dq layers for the output and loss are separated 
                q_node_output = fc_out_node.args[0]
                assert len(q_node_output.users) == 2, print("the q node does not have two outputs, which should be the general behaviour")
                for user in q_node_output.users:
                    if user != fc_out_node:
                        loss_node = next(loss_user for loss_user in user.users)
                
            new_node = nn.Identity()
            new_node_name = 'replaced_loss'
            model.add_module(new_node_name, new_node)
            with model.graph.inserting_before(loss_node):
                args = []
                for arg in loss_node.args:
                    if type(arg) == fx.Node:
                        if arg.op != "get_attr":
                            args.append(arg)
                new_node = model.graph.call_module(new_node_name, tuple(args),{})
                ptr = loss_node
                while ptr != loss_end_node:
                    ptr.replace_all_uses_with(new_node)
                    temp=ptr.next
                    model.graph.erase_node(ptr)
                    ptr=temp
                
                ptr.replace_all_uses_with(new_node)
                model.graph.erase_node(loss_end_node)
    
    model.graph.lint()
    model.recompile()
    
    return model


def _bias_calibration_hook(m, x, y, calibration_factor, bias_module):
    bias_error = 0
    if isinstance(x, tuple):
        x = x[0]
    if len(x.shape) == 3:
        float_mean = x.mean(dim=(0,1))
        quant_mean = y.mean(dim=(0,1))
        bias_error = float_mean - quant_mean 
    elif len(x.shape) == 4:
        if x.shape[1] == bias_module.shape[0]:
            float_mean = x.mean(dim=(0,2,3))
            quant_mean = y.mean(dim=(0,2,3)) 
            bias_error = float_mean - quant_mean                                          
        elif x.shape[3] == bias_module.shape[0]:
            float_mean = x.mean(dim=(0,1,2))    
            quant_mean = y.mean(dim=(0,1,2))  
            bias_error = float_mean - quant_mean 

    bias_module.data += (bias_error * calibration_factor)
    return y


def add_bias_calibration_hook(model, calibration_factor=0):
    all_hooks = []    
    module_partitions = get_source_partitions(
        model.graph, [torch.nn.Linear, torch.nn.functional.linear, torch.nn.Conv2d, torch.nn.functional.conv2d]
    )

    for module_or_fn_type, partitions in module_partitions.items():
        for p in partitions:
            bias_node = None
            for param_node in p.params:
                weight_or_bias = getattr(model, param_node.target)
                if weight_or_bias.ndim == 1:  # type: ignore[attr-defined]
                    bias_node = param_node
                #
            #
            if bias_node is not None:
                bias_module = getattr(model, bias_node.target)
            else:
                continue
            
            output_node = None
            for out_node in p.output_nodes:
                if out_node.target in [torch.ops.aten.convolution.default, torch.ops.aten.addmm.default, 
                                       torch.ops.aten.view.default, torch.ops.aten.add.Tensor]:
                    output_node = out_node
                    break
                #
            #
            if output_node is None:
                raise ValueError(
                    "Could not find an user of act node for conv within matched pattern."
                )
            #
            fake_quantize_module = getattr(model, output_node.next.target)
            
            _bias_calibration_hook_binded = partial(_bias_calibration_hook, \
                calibration_factor=calibration_factor, bias_module=bias_module)
            this_hook = fake_quantize_module.register_forward_hook(_bias_calibration_hook_binded)
            all_hooks.append(this_hook)
        #
    #
    return all_hooks


def _fc_outlier_supression_hook(m, x):
    if isinstance(x, tuple):
        x = x[0]
    mean_val = x.mean(dim=(0,1))
    std_val = x.std(dim=(0,1))
    clip_val_max = mean_val + 3*std_val
    clip_val_min = mean_val - 3*std_val
    # clip_val_max = mean_val - 3*std_val
    # clip_val_min = mean_val + 3*std_val
    x = torch.clip(x, min=clip_val_min, max = clip_val_max)
    return tuple([x])


def is_mlp_fc2_layer(node, find_level, found_gelu=False, gelu_node=None):
    if find_level < 0:
        return False, gelu_node
    if node.target is torch.ops.aten.addmm.default:
        if found_gelu: 
            return True, gelu_node
        # else: 
        #     # found linear before the gelu layer
        #     return False, gelu_node
        #
    #
    elif node.target is torch.ops.aten.gelu.default:
        gelu_node = node
        found_gelu = True
        
    if hasattr(node, "args") and len(node.args)>0:
        return is_mlp_fc2_layer(node.args[0], find_level-1, found_gelu, gelu_node)
    else:
        return False, gelu_node
    #


def add_fc_outlier_supression_hook(model):
    all_modules = dict(model.named_modules())
    all_hooks = []
    
    module_partitions = get_source_partitions(
        model.graph, [torch.nn.Linear, torch.nn.functional.linear]
    )
    
    for module_or_fn_type, partitions in module_partitions.items():
        for p in partitions:
            inp_nodes = p.input_nodes 
            prev_node = None
            for node in inp_nodes:
                if isinstance(node.prev.target, str) and 'param_constant' in node.prev.target:
                    continue
                else:
                    prev_node = node
        
            assert prev_node is not None, print("prev_node is node in trying to iterate over nodes") 
            
            found_mlp_fc2_layer, gelu_node = is_mlp_fc2_layer(prev_node, 6)
            if found_mlp_fc2_layer:
                output_node = None
                for node in p.output_nodes:
                    if not(isinstance(node.target,str) and 'param_constant' in node.target):
                        output_node = node 
                        break
                if output_node is not None:
                    if isinstance(output_node.target,str) and hasattr(model, output_node.target) and isinstance(getattr(model, output_node.target), torch.nn.Module):
                        act_node = output_node
                    else:
                        act_node = output_node.next
                this_hook1 = getattr(model, act_node.target).register_forward_pre_hook(_fc_outlier_supression_hook)
                # this_hook2 = getattr(model, gelu_node.next.target).register_forward_pre_hook(_fc_outlier_supression_hook)
                all_hooks.append(this_hook1)
                # all_hooks.append(this_hook2)
                
    return all_hooks