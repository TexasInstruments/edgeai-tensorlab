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

def add_constant_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 0, f'input {node.inputs} of {node.name} has more than 0 inputs. Constant node should have 0 inputs'
    if (sp_val:= node.attrs.get('sparse_value',None)) is not None:
        if isinstance(sp_val, gs.Constant):
            sp_val = sp_val.values
        try:
            val = torch.nn.Parameter(torch.tensor(sp_val))
        except:
            val = torch.tensor(sp_val)
    elif (val:= node.attrs.get('value',None)) is not None:
        if isinstance(val, gs.Constant):
            val = val.values
        try:
            val = torch.nn.Parameter(torch.tensor(val))
        except:
            val = torch.tensor(val)
    elif (val:= node.attrs.get('value_float',None)) is not None:
        val = torch.nn.Parameter(torch.tensor(val))
    elif (val:= node.attrs.get('value_int',None)) is not None:
        val = (torch.tensor(val))
    elif (val:= node.attrs.get('value_floats',None)) is not None:
        val = torch.nn.Parameter(torch.tensor(val))
    elif (val:= node.attrs.get('value_ints',None)) is not None:
        val = (torch.tensor(val))
    elif (val:= node.attrs.get('value_string',None)) is not None:
        val = val
    elif (val:= node.attrs.get('value_strings',None)) is not None:
        val = val
    else:
        raise ValueError(f'node {node.name} has no value')
    if isinstance(val, torch.nn.Parameter):
        torch_module.register_parameter(node.name, val)
    elif isinstance(val, torch.Tensor):
        torch_module.register_buffer(node.name, val)
    else:
        setattr(torch_module, node.name, val)
    
    torch_nodes[node.name] = torch_graph.get_attr(node.name)

def torch_costant_of_shape(shape, value=0.0):
    device = 'cpu'
    if isinstance(shape, torch.Tensor):
        device = shape.device
        shape = shape.tolist()
    return torch.ones(shape).to(device)*value

def add_constant_of_shape_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 'value' in node.attrs, f'node {node.name} with op {node.op} has no value'
    value = node.attrs.get('value')
    if isinstance(value, gs.Constant):
        value = value.values.tolist()[0]
    shape = utils.get_input_from_node(node.inputs[0], torch_graph, torch_nodes, torch_module, list)
    if state.module_based:
        args = [shape]
        module = utils.WrappedModule(node.op, torch_module, torch_costant_of_shape, args, dict(value=value))
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_costant_of_shape, (shape,), dict(value=value), name=node.name)

def torch_eye_like(inp, dtype=torch.float,k=0 ):
    assert (inp.dim() if isinstance(inp, torch.Tensor) else len(inp))== 2, f'eye_like only support 2D tensor, but got {inp.dim()if isinstance(inp, torch.Tensor) else len(inp)}D'
    x = torch.zeros_like(inp, dtype=dtype)
    for i in range(min(inp.shape[0],inp.shape[1])):
        x[i,i+k] = 1
    return x

def add_eye_like_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    dtype = node.attrs.get('dtype',utils.TensorProto.FLOAT)
    dtype = utils.onnx_2_torch_type_mapping[dtype]
    k = node.attrs.get('k',0)
    inp = utils.get_input_from_node(node.inputs[0], torch_graph, torch_nodes, torch_module, torch.nn.Parameter if inp.shape else torch.Tensor)
    if state.module_based:
        module = utils.WrappedModule(node.op, torch_module, torch_eye_like, args, dict(dtype=dtype,k=k))
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_eye_like, (inp,), dict(dtype=dtype,k=k), name=node.name)