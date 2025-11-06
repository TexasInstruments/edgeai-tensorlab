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
from numbers import Number
import numpy as np

def torch_concat(tensors, dim=0):
    if all(isinstance( t, torch.Tensor) for t in tensors):
        return torch.concatenate(tensors, dim)
    if any (isinstance( t, (list, tuple)) for t in tensors):
        tensors = [t.cpu() if isinstance( t, torch.Tensor) else t for t in tensors]
        tensors = [t.tolist() if isinstance( t, (torch.Tensor, np.ndarray)) else t for t in tensors]
        tensors = [torch.tensor(t) for t in tensors]
        return torch.concat(tensors, axis=dim)
    raise NotImplementedError


def add_concat_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    dim = node.attrs.get('axis')
    if state.module_based:
        module = utils.WrappedModule(node.name, node.op, torch_module, torch_concat, kwargs=dict(dim=dim))
        torch_module.add_module(node.name, module)
        # args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple([args]))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_concat, tuple([args]),  dict(dim=dim), name=node.name)

def torch_stack(tensors, dim=0):
    if all(isinstance( t, torch.Tensor) for t in tensors):
        return torch.stack(tensors, dim)
    if any (isinstance( t, (list, tuple)) for t in tensors):
        tensors = [t.cpu() if isinstance( t, torch.Tensor) else t for t in tensors]
        tensors = [t.tolist() if isinstance( t, (torch.Tensor, np.ndarray)) else t for t in tensors]
        tensors = [torch.tensor(t) for t in tensors]
        return torch.stack(tensors, axis=dim)
    raise NotImplementedError

def add_concat_from_sequence_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    dim = node.attrs.get('axis')
    new_axis = node.attrs.get('new_axis', 0) == 1
    func = torch_stack if new_axis else torch_concat
    if state.module_based:
        module = utils.WrappedModule(node.name, node.op, torch_module, func, kwargs=dict(dim=dim))
        torch_module.add_module(node.name, module)
        # args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple([args]))
    else:
        torch_nodes[node.name] = torch_graph.call_function(func, tuple([args]),  dict(dim=dim), name=node.name)

def torch_expand(x:torch.Tensor, shape):
    if isinstance(shape, torch.Tensor):
        shape = shape.int().tolist()
    return x*torch.ones(shape).to(x.device)

def add_expand_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    if state.module_based:
        module = utils.WrappedModule(node.name, node.op, torch_module, torch_expand, args, )
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_expand, tuple(args),   name=node.name)


def torch_split(input, split, dim=0):
    if isinstance(split, torch.Tensor):
        split = split.int().tolist()
    if isinstance(split, (list, tuple)):
        if len(split) == 1:
            return input
        return torch.split(input, split, dim)
    if isinstance(split, int):
        if split == 1:
            return input
        return torch.split(input, split, dim)
    raise NotImplementedError

def add_split_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 1<=len(node.inputs)<=2, f'{node.name} with operator {node.op} should have 1 or 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    kwargs = dict(dim = node.attrs.get('axis',0))
    if 'split' in node.attrs:
        split= node.attrs.get('split')
        args.append(split)
    if 'num_outputs' in node.attrs:
        split= node.attrs.get('num_outputs')
        args.append(split) if len(args)<2 else None
    if state.module_based:
        module = utils.WrappedModule(node.name, node.op, torch_module, torch_split, args, kwargs)
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_split, tuple(args),  kwargs, name=node.name)

def torch_tile(input, dims):
    if isinstance(dims, torch.Tensor):
        dims = tuple(dims.int().tolist())
    return torch.tile(input, dims)

def add_tile_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    if state.module_based:
        module = utils.WrappedModule(node.name, node.op, torch_module, torch_tile, args)
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_tile, tuple(args),  name=node.name)

def add_split_to_sequence_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")

def add_reverse_sequence_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")

def add_sequence_at_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")

def add_sequence_construct_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")

def add_sequence_empty_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")

def add_sequence_erase_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")

def add_sequence_insert_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")

def add_sequence_length_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")

def add_sequence_map_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")
