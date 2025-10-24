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
from operator import setitem
from . import utils
import numpy as np

def torch_scatter(x: torch.Tensor, indices, updates, axis=0):
    output = torch.scatter(x,axis, indices, updates)
    return output
def add_scatter_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 3, f'{node.name} with operator {node.op} should have 3 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.Tensor, torch.Tensor]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    axis = node.attrs.get('axis', 0)
    if state.module_based:
        module = utils.WrappedModule(node.op, torch_module, torch_scatter, args, dict(axis=axis),)
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_scatter, tuple(args),  dict(axis=axis), name=node.name)

def torch_scatter_elements(x:torch.Tensor, indices:torch.Tensor, updates:torch.Tensor, axis=0, reduce='none'):
    # Use torch.scatter which is the direct equivalent
    if reduce is None or reduce == 'none':
        return torch.scatter(x, axis,indices, updates)
    return torch.scatter(x, axis,indices, updates, reduce=reduce)

def add_scatter_elements_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 3, f'{node.name} with operator {node.op} should have 3 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.Tensor, torch.Tensor]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    axis = node.attrs.get('axis', 0)
    reduce = node.attrs.get('reduction', 'none')
    if state.module_based:
        module = utils.WrappedModule(node.op, torch_module, torch_scatter_elements, args, dict(axis=axis, reduce=reduce))
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_scatter_elements, tuple(args),  dict(axis=axis, reduce=reduce), name=node.name)

reduce_func_dict = {
    'mul': torch.mul,
    'min': torch.min,
    'max': torch.max,
}

def torch_np_index(shape):
    result = []
    for np_index in np.ndindex(shape):
        result.append(np_index)
    return torch.tensor(result)

def torch_scatter_nd(data:torch.Tensor, indices:torch.Tensor, updates:torch.Tensor, reduce='none'):
    # Handle batch dimensions
    output = data.clone()
    indices = tuple([indices[..., i].long() for i in range(indices.shape[-1])])
    
    # Apply the update
    if reduce is None or reduce == 'none':
        output.index_put_(indices, updates)
    elif reduce == 'add':
        output.index_put_(indices, updates, True)
    elif reduce in reduce_func_dict:
        updates_indices = torch_np_index(indices.shape[:-1]).to(output)
        for idx in updates_indices:
            output[indices[idx]] = reduce_func_dict[reduce](output[indices[idx]], updates[idx])
    else:
        raise ValueError(f'Reduction {reduce} not supported')
    return output
def add_scatter_nd_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 3, f'{node.name} with operator {node.op} should have 3 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, torch.Tensor, torch.Tensor]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    reduce = node.attrs.get('reduction', 'none')
    if state.module_based:
        module = utils.WrappedModule(node.op, torch_module, torch_scatter_nd, args, dict(reduce=reduce))
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_scatter_nd, tuple(args),  dict(reduce=reduce), name=node.name)

def add_tensor_scatter_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise ValueError(f'{node.name} with operator {node.op} is not implemented')