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
# from operator import getitem
from . import utils
import numpy as np
import copy

def torch_gather(x, indices, axis=0):
    if axis < 0:
        axis += x.dim() if isinstance(x, torch.Tensor) else len(x)
    if  isinstance(indices, torch.Tensor) and indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        indices = indices.to(torch.int64)
    if (isinstance(indices, torch.Tensor) and  not indices.shape):
        indices = indices.tolist()
    if isinstance(indices, int):
        return torch.select(x, axis, indices)
    if isinstance(indices, (list, tuple)):
        indices = torch.tensor(indices).to(x.device)
    if isinstance(indices, torch.Tensor ):
        indices_dim = indices.dim()
        indices_shape = indices.shape
        indices = indices.reshape(-1)
        if indices.dim() == 1:
            indices = torch.where(indices<0, indices+x.shape[axis], indices)
            result =  torch.index_select( x, axis, indices)
        else:
            raise NotImplementedError
        if indices_dim != 1:
            shape = result.shape[:axis] + indices_shape + result.shape[axis+1:]
            result = result.reshape(shape)
        return result


def add_gather_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.Tensor]
    args = [utils.get_input_from_node(node, inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    if args[1].op == 'get_attr':
        args[1] = getattr(torch_module, args[1].target)
        if isinstance(args[1], torch.Tensor) and args[1].dim() == 0:
            args[1] = args[1].cpu().tolist() # TODO: TBD either to make it fixed
    axis = node.attrs.get('axis', 0)
    if state.module_based:
        module = utils.WrappedModule(node.name, node.op, torch_module, torch_gather, args, dict(axis=axis),)
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_gather, tuple(args),  dict(axis=axis), name=node.name)

def torch_gather_elements(x:torch.Tensor, indices:torch.Tensor, axis=0):
    # Handle negative axis
    if axis < 0:
        axis = (x.dim() if isinstance(x, torch.Tensor) else len(x)) + axis
        
    # Use torch.gather which is the direct equivalent
    return torch.gather(x, dim=axis, index=indices)

def add_gather_elements_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.Tensor]
    args = [utils.get_input_from_node(node, inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    axis = node.attrs.get('axis', 0)
    if state.module_based:
        module = utils.WrappedModule(node.name, node.op, torch_module, torch_gather_elements, args, dict(axis=axis),)
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_gather_elements, tuple(args),  dict(axis=axis), name=node.name)

def torch_gather_nd(data:torch.Tensor, indices:torch.Tensor, batch_dims=0):
    # Handle batch dimensions
    if batch_dims > 0:
        batch_shape = indices.shape[:batch_dims]
        
        # Reshape to combine batch dimensions
        batch_size = torch.prod(torch.tensor(batch_shape)).item()
        data_reshape = data.reshape(batch_size, *data.shape[batch_dims:])
        indices_reshape = indices.reshape(batch_size, *indices.shape[batch_dims:])
        
        # Process each batch element
        result = []
        for i in range(batch_size):
            # Recursive call without batch dimensions
            result.append(torch_gather_nd(data_reshape[i], indices_reshape[i]))
        
        # Reshape back to original batch dimensions
        return torch.stack(result).reshape(*batch_shape, *result[0].shape)
    
    # Convert indices to tuple of indices for each dimension
    index_tuples = []
    for dim in range(indices.shape[-1]):
        index_tuples.append(indices[..., dim])
    
    # Use basic indexing to gather values
    return data[tuple(index_tuples)]

def add_gather_nd_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.Tensor]
    args = [utils.get_input_from_node(node, inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    batch_dims = node.attrs.get('batch_dims', 0)
    if state.module_based:
        module = utils.WrappedModule(node.name, node.op, torch_module, torch_gather_nd, args, dict(batch_dims=batch_dims),)
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_gather_nd, tuple(args),  dict(batch_dims=batch_dims), name=node.name)