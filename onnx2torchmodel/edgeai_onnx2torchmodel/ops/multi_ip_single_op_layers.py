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


import onnx_graphsurgeon as gs
import torch
from . import utils
import operator

onnx_to_torch = {
    'Add': operator.add,
    'Mul': operator.mul,
    'Sub': operator.sub,
    'Div': operator.truediv,
    'And': torch.logical_and,
    'Or': torch.logical_or,
    'Xor': torch.logical_xor,
    'BitwiseAnd': torch.bitwise_and,
    'BitwiseOr': torch.bitwise_or,
    'BitwiseXor': torch.bitwise_xor,
    'Equal': operator.eq,
    'Greater': operator.gt,
    'Less': operator.lt,
    'GreaterOrEqual': operator.ge,
    'LessOrEqual': operator.le,
    'Pow': operator.pow,
}

def wrap_for_tensor(fn):
    def wrapped(x,y):
        args = [x,y]
        device = [a.device for a in args if isinstance(a, torch.Tensor)]
        if len(device) == 1:
            args = [a if isinstance(a, torch.Tensor) else torch.tensor(a, device=device[0]) for a in args]
        x,y = args
        return fn(x,y)
    wrapped.orig_func = fn
    return wrapped

def add_node_2_torch_graph_multi_ip_1op(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    if node.op in onnx_to_torch:
        assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
        types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
        args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
        func = onnx_to_torch[node.op]
        # if func.__module__ == 'torch':
        func = wrap_for_tensor(func)

        if state.module_based:
            module = utils.WrappedModule(node.name, node.op, torch_module, func, args)
            torch_module.add_module(node.name, module)
            args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]

            torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
        else:
            torch_nodes[node.name] = torch_graph.call_function(func, tuple(args),  name=node.name)
    else:
        raise NotImplementedError (f"{node.name} with operator {node.op} is not implemented")


def add_mod_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    fmod = node.attrs.get('fmod', 0)==1
    if fmod:
        func = torch.fmod
    else:  
        func = torch.remainder
    if state.module_based:
        module = utils.WrappedModule(node.name, node.op, torch_module, func, args)
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(func, tuple(args), dict(fmod=fmod), name=node.name)

def add_sum_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    if state.module_based:
        module = torch.sum
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch.sum, tuple(args),  name=node.name)