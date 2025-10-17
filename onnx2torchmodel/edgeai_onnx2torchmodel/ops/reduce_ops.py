# Copyright (c) 2018-2025, Texas Instruments
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


import torch
import onnx_graphsurgeon as gs
from . import utils

#TODO add support for noop_with_empty_axes
def torch_reduce_max(x, axes=None, keepdims=True, noop_with_empty_axes = False):
    return torch.amax(x, dim=axes, keepdim=keepdims)

def add_reduce_max_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 1<=len(node.inputs)<= 2, f'{node.name} with operator {node.op} should have between 1 and 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter , list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    
    keepdims = node.attrs.get('keepdims', 0) == 1
    noop_with_empty_axes  = node.attrs.get('noop_with_empty_axes', 0) == 1
    kwargs = dict(keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes)
    if 'axes' in node.attrs:
        axes = node.attrs['axes'] 
        kwargs['axes'] = axes
    torch_nodes[node.name] = torch_graph.call_function(torch_reduce_max, tuple(args),  kwargs, name=node.name)

def torch_reduce_min(x, axes=None, keepdims=True, noop_with_empty_axes = False):
    return torch.amin(x, dim=axes, keepdim=keepdims)
def add_reduce_min_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 1<=len(node.inputs)<= 2, f'{node.name} with operator {node.op} should have between 1 and 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter , list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    
    keepdims = node.attrs.get('keepdims', 0) == 1
    noop_with_empty_axes  = node.attrs.get('noop_with_empty_axes', 0) == 1
    kwargs = dict(keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes)
    if 'axes' in node.attrs:
        axes = node.attrs['axes'] 
        kwargs['axes'] = axes
    torch_nodes[node.name] = torch_graph.call_function(torch_reduce_min, tuple(args),  kwargs, name=node.name)

def torch_reduce_mean(x, axes, keepdims=True, noop_with_empty_axes = False):
    return torch.mean(x, dim=axes, keepdim=keepdims)

def add_reduce_mean_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 1<=len(node.inputs)<= 2, f'{node.name} with operator {node.op} should have between 1 and 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter , list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    
    keepdims = node.attrs.get('keepdims', 0) == 1
    noop_with_empty_axes  = node.attrs.get('noop_with_empty_axes', 0) == 1
    kwargs = dict(keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes)
    if 'axes' in node.attrs:
        axes = node.attrs['axes'] 
        kwargs['axes'] = axes
    torch_nodes[node.name] = torch_graph.call_function(torch_reduce_mean, tuple(args),  kwargs, name=node.name)

def torch_reduce_l1(x, axes, keepdims=True, noop_with_empty_axes = False):
    if axes is None:
        # Reduce over all dimensions
        result = torch.norm(x, p=1)
        if keepdims:
            result = result.view([1] * x.dim())
    else:
        # Convert to tuple if it's a list
        if isinstance(axes, torch.Tensor):
            axes = axes.tolist()
        if isinstance(axes, list):
            axes = tuple(axes)
        result = torch.norm(x, p=1, dim=axes, keepdim=keepdims)
    return result
    
def add_reduce_l1_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 1 <= len(node.inputs) <= 2, f'{node.name} with operator {node.op} should have between 1 and 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter , list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    keepdims = node.attrs.get('keepdims', 0) == 1
    noop_with_empty_axes  = node.attrs.get('noop_with_empty_axes', 0) == 1
    kwargs = dict(keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes)
    if 'axes' in node.attrs:
        axes = node.attrs['axes'] 
        kwargs['axes'] = axes
    torch_nodes[node.name] = torch_graph.call_function(torch_reduce_l1, tuple(args),  kwargs, name=node.name) 

def torch_reduce_l2(x, axes, keepdims=True, noop_with_empty_axes = False):
    if axes is None:
        # Reduce over all dimensions
        result = torch.norm(x, p=2)
        if keepdims:
            result = result.view([1] * x.dim())
    else:
        # Convert to tuple if it's a list
        if isinstance(axes, torch.Tensor):
            axes = axes.tolist()
        if isinstance(axes, list):
            axes = tuple(axes)
        result = torch.norm(x, p=2, dim=axes, keepdim=keepdims)
    return result

def add_reduce_l2_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 1 <= len(node.inputs) <= 2, f'{node.name} with operator {node.op} should have between 1 and 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter , list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    keepdims = node.attrs.get('keepdims', 0) == 1
    noop_with_empty_axes  = node.attrs.get('noop_with_empty_axes', 0) == 1
    kwargs = dict(keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes)
    if 'axes' in node.attrs:
        axes = node.attrs['axes'] 
        kwargs['axes'] = axes
    torch_nodes[node.name] = torch_graph.call_function(torch_reduce_l2, tuple(args),  kwargs, name=node.name) 

def torch_reduce_sum(x, axes, keepdims=True, noop_with_empty_axes = False):
    if axes is None:
        # Reduce over all dimensions
        result = torch.sum(x)
        if keepdims:
            result = result.view([1] * x.dim())
    else:
        # Convert to tuple if it's a list
        if isinstance(axes, torch.Tensor):
            axes = axes.tolist()
        if isinstance(axes, list):
            axes = tuple(axes)
        result = torch.sum(x, dim=axes, keepdim=keepdims)
    return result

def add_reduce_sum_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 1 <= len(node.inputs) <= 2, f'{node.name} with operator {node.op} should have between 1 and 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter , list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    keepdims = node.attrs.get('keepdims', 0) == 1
    noop_with_empty_axes  = node.attrs.get('noop_with_empty_axes', 0) == 1
    kwargs = dict(keepdims=keepdims, noop_with_empty_axes=noop_with_empty_axes)
    if 'axes' in node.attrs:
        axes = node.attrs['axes'] 
        kwargs['axes'] = axes
    torch_nodes[node.name] = torch_graph.call_function(torch_reduce_l2, tuple(args),  kwargs, name=node.name) 

def add_reduce_log_sum_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")

def add_reduce_log_sum_exp_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")

def add_reduce_prod_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")

def add_reduce_sum_square_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")
