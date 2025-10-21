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

def add_celu_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    alpha = node.attrs.get('alpha', 1.0)
    if state.module_based:
        module = torch.nn.CELU(alpha=alpha)
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.celu, tuple(args), dict(alpha=alpha), name=node.name)

def add_elu_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    alpha = node.attrs.get('alpha', 1.0)
    if state.module_based:
        module = torch.nn.ELU(alpha=alpha)
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.elu, tuple(args), dict(alpha=alpha), name=node.name)

def torch_hardsigmoid(x, alpha=0.2, beta=0.5):
    return torch.clip(alpha*x+beta, 0, 1)

def add_hardsigmoid_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    alpha = node.attrs.get('alpha', 0.5)
    beta = node.attrs.get('beta', 0.5)
    if state.module_based:
        if alpha == 1/6 and beta == 1/2:
            module = torch.nn.Hardsigmoid()
        else:
            module = utils.WrappedModule(node.op, torch_module, torch_hardsigmoid, args, dict(alpha=alpha, beta=beta), )
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_hardsigmoid, tuple(args), dict(alpha=alpha, beta=beta), name=node.name)

def torch_hardswish(x):
    return x*torch_hardsigmoid(x, alpha=1/6, beta=0.5)

def add_hardswish_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    if state.module_based:
        module = torch.nn.Hardswish()
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_hardswish, tuple(args), name=node.name)

def torch_hardmax(input_tensor, axis=-1):
    if axis < 0:
        axis = input_tensor.dim() + axis
    
    # Find indices of maximum values along the specified axis
    max_values, _ = torch.max(input_tensor, dim=axis, keepdim=True)
    
    # Create a mask where elements equal the maximum along the axis
    max_mask = (input_tensor == max_values).to(torch.float32)
    
    # Use cumulative sum to identify the first occurrence of maximum
    # The first max will have a value of 1, and subsequent max values will be > 1
    cumsum = torch.cumsum(max_mask, dim=axis)
    
    # First max has cumsum=1, so we can create the hardmax mask
    hardmax_mask = (cumsum == 1) & (max_mask == 1)
    
    return hardmax_mask.to(torch.float32)

def add_hardmax_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    axis = node.attrs.get('axis', -1)
    if state.module_based:
        module = utils.WrappedModule(node.op, torch_module, torch_hardmax, args, dict(axis=axis),)
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_hardmax, tuple(args), dict(axis=axis), name=node.name)

def add_leakyrelu_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    alpha = node.attrs.get('alpha', 0.01)
    if state.module_based:
        module = torch.nn.LeakyReLU(negative_slope=alpha)
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.leaky_relu, tuple(args), dict(negative_slope=alpha), name=node.name)

def add_log_softmax_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    axis = node.attrs.get('axis', -1)
    if state.module_based:
        module = torch.nn.LogSoftmax(dim=axis)
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch.log_softmax, tuple(args), dict(dim=axis), name=node.name)

def add_mish_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    if state.module_based:
        module = torch.nn.Mish()
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.mish, tuple(args), name=node.name)

def add_relu_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    if state.module_based:
        module = torch.nn.ReLU()
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch.relu, tuple(args), name=node.name)

def add_selu_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    alpha = node.attrs.get('alpha', 1.6732632423543772)
    gamma = node.attrs.get('gamma', 1.0507009873554805)
    assert alpha == 1.6732632423543772 and gamma == 1.0507009873554805, f'{node.name} with operator {node.op} should have alpha=1.6732632423543772 and gamma=1.0507009873554805, but got alpha={alpha} and gamma={gamma}'
    if state.module_based:
        module = torch.nn.SELU()
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.selu, tuple(args), name=node.name)

def add_sigmoid_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    if state.module_based:
        module = torch.nn.Sigmoid()
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch.sigmoid, tuple(args), name=node.name)

def add_softmax_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    axis = node.attrs.get('axis', -1)
    if state.module_based:
        module = torch.nn.Softmax(dim=axis)
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch.softmax, tuple(args), dict(dim=axis), name=node.name)

def torch_swish(x, alpha=1.0):
    return x * torch.sigmoid(x * alpha)

def add_swish_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    alpha = node.attrs.get('alpha', 1.0)
    if state.module_based:
        module = utils.WrappedModule(node.op, torch_module, torch_swish, args, dict(alpha=alpha), )
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_swish, tuple(args),dict(alpha=alpha), name=node.name)

def add_prelu_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    if state.module_based:
        module = torch.nn.PReLU()
        weight = args[1]
        if args[1].op == 'get_attr':
            args = args[0:1]
            module.weight = getattr(torch_module, weight.target)
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.prelu, tuple(args), name=node.name)

def add_softplus_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    if state.module_based:
        module = torch.nn.Softplus()
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.softplus, tuple(args), name=node.name)

def add_softsign_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    if state.module_based:
        module = torch.nn.Softsign()
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.softsign, tuple(args), name=node.name)

def add_thresholded_relu_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    alpha = node.attrs.get('alpha', 1.0)
    if state.module_based:
        module = torch.nn.Threshold(alpha,0)
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.threshold, tuple(args, alpha,0), name=node.name)

def add_trilu_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter, list]
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    if len(args)<2:
        args = args[0],0
    upper = node.attrs.get('upper', 1)==1
    func = torch.triu if upper else torch.tril
    torch_nodes[node.name] = torch_graph.call_function( func, tuple(args), name=node.name)
    if state.module_based:
        module = utils.WrappedModule(node.op, torch_module, func, args )
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch.tril, tuple(args), name=node.name)