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
from operator import getitem
from . import utils
import numpy as np

def torch_gather(x, indices, axis=0):
    if isinstance(indices, torch.Tensor) and indices.shape:
        return torch.index_select(x, axis, indices)
    slices = [slice(None) for _ in range(x.ndim)]
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu()
        indices = indices.numpy()
    if isinstance(indices, np.ndarray):
        indices = indices.tolist()
        
    if axis < 0:
        axis += x.ndim
    if indices<0:
        indices += x.shape[axis]
    slices[axis] = indices
    return getitem(x, tuple(slices))

def add_gather_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.Tensor]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    axis = node.attrs.get('axis', 0)
    torch_nodes[node.name] = torch_graph.call_function(torch_gather, tuple(args),  dict(axis=axis), name=node.name)

def torch_gather_elements(x:torch.Tensor, indices:torch.Tensor, axis=0):
    # Handle negative axis
    if axis < 0:
        axis = x.dim() + axis
        
    # Use torch.gather which is the direct equivalent
    return torch.gather(x, dim=axis, index=indices)

def add_gather_elements_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.Tensor]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    axis = node.attrs.get('axis', 0)
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
    return data[index_tuples]

def add_gather_nd_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.Tensor]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    batch_dims = node.attrs.get('batch_dims', 0)
    torch_nodes[node.name] = torch_graph.call_function(torch_gather_nd, tuple(args),  dict(batch_dims=batch_dims), name=node.name)