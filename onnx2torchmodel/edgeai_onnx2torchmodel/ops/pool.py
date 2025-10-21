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
from .pad import Pad 

def add_avg_pool_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    #TODO auto_pad setting
    pad_mode = node.attrs.get('auto_pad', 'NOTSET') 
    if pad_mode == 'SAME_UPPER':
        pass
    elif pad_mode == 'SAME_LOWER':
        pass
    elif pad_mode == 'VALID':
        pass
    
    kernel_size = node.attrs.get('kernel_shape')
    stride = node.attrs.get('strides')
    padding = node.attrs.get('pads', [0]*(2*len(kernel_size)))
    ceil_mode = node.attrs.get('ceil_mode', 0) == 1
    count_include_pad = node.attrs.get('count_include_pad', 0) == 1

    kernel_size = tuple(kernel_size)
    add_padding = False
    if padding:
        if padding[:len(kernel_size)] == padding[len(kernel_size):]:
            padding = padding[:len(kernel_size)]
        else:
            add_padding = True
            old_padding = padding
            padding = [0]*len(kernel_size)
    else:
        padding = [0]*len(kernel_size)
    
    kwargs = dict(
        kernel_size = kernel_size,
        stride = stride,
        padding = padding,
        ceil_mode = ceil_mode,
        count_include_pad = count_include_pad
    )
    
    if len(kernel_size) == 1:
        func = torch.nn.functional.avg_pool1d
    if len(kernel_size) == 2:
        func = torch.nn.functional.avg_pool2d
    if len(kernel_size) == 3:
        func = torch.nn.functional.avg_pool3d
    if add_padding and padding:
            pad_module = Pad(old_padding)
            torch_module.add_module(node.name+'_pad', pad_module)
            padding_node = torch_graph.call_module(node.name+'_pad', tuple(args[0:1]))
            args[0] = padding_node
    if state.module_based:
        module = utils.WrappedModule(node.op, torch_module, func, args, kwargs)
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function( func, tuple(args),  kwargs, name=node.name)
    for attr in node.attrs:
        if attr in kwargs:
            continue
        torch_nodes[node.name].meta[attr] = node.attrs[attr]


def add_max_pool_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    #TODO auto_pad setting
    pad_mode = node.attrs.get('auto_pad', 'NOTSET') 
    if pad_mode == 'SAME_UPPER':
        pass
    elif pad_mode == 'SAME_LOWER':
        pass
    elif pad_mode == 'VALID':
        pass
    
    kernel_size = node.attrs.get('kernel_shape')
    stride = node.attrs.get('strides')
    padding = node.attrs.get('pads', [0]*(2*len(kernel_size)))
    ceil_mode = node.attrs.get('ceil_mode', 0) == 1
    dilation = node.attrs.get('dilations', [1]*len(kernel_size))
    storage_order = node.attrs.get('storage_order', 0) == 1
    return_indices = len(node.outputs) == 2

    kernel_size = tuple(kernel_size)
    add_padding =False
    if padding:
        if padding[:len(kernel_size)] == padding[len(kernel_size):]:
            padding = padding[:len(kernel_size)]
        else:
            add_padding = True
            old_padding = padding
            padding = [0]*len(kernel_size)
    else:
        padding = [0]*len(kernel_size)
    
    kwargs = dict(
        kernel_size = kernel_size,
        stride = stride,
        padding = padding,
        ceil_mode = ceil_mode,
        dilation=dilation,
        return_indices = return_indices
    )
    
    if len(kernel_size) == 1:
        func = torch.nn.functional.max_pool1d
    if len(kernel_size) == 2:
        func = torch.nn.functional.max_pool2d
    if len(kernel_size) == 3:
        func = torch.nn.functional.max_pool3d
    if add_padding and padding:
            pad_module = Pad(old_padding)
            torch_module.add_module(node.name+'_pad', pad_module)
            padding_node = torch_graph.call_module(node.name+'_pad', tuple(args[0:1]))
            args[0] = padding_node
    
    if state.module_based:
        module = utils.WrappedModule(node.op, torch_module, func, args, kwargs)
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function( func, tuple(args),  kwargs, name=node.name)
    for attr in node.attrs:
        if attr in kwargs:
            continue
        torch_nodes[node.name].meta[attr] = node.attrs[attr]

def torch_global_avg_pool(x:torch.Tensor):
    if x.dim() == 3:
        func = torch.nn.functional.adaptive_avg_pool1d
    elif x.dim() == 4:
        func = torch.nn.functional.adaptive_avg_pool2d
    elif x.dim() == 5:
        func = torch.nn.functional.adaptive_avg_pool3d
    else: 
        raise NotImplementedError('global_avg_pool only supports 3d, 4d and 5d inputs but got {}D'.format(x.dim()))
    return func(x, 1)


def add_global_avg_pool_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]

    if state.module_based:
        module = utils.WrappedModule(node.op, torch_module, torch_global_avg_pool, args, )
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function( torch_global_avg_pool, tuple(args), name=node.name)

def add_global_max_pool_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f'{node.name} with operator {node.op} is not implemented')

def add_global_lp_pool_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f'{node.name} with operator {node.op} is not implemented')

def add_lp_pool_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f'{node.name} with operator {node.op} is not implemented')

def add_max_roi_pool_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f'{node.name} with operator {node.op} is not implemented')

def add_max_unpool_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f'{node.name} with operator {node.op} is not implemented')



