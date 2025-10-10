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

def torch_pad(x, pad, value=0.0, axes=None, mode='constant',):
    axes = axes or list(range(x.ndim))
    t_axes = list(set(axes))
    assert len(t_axes) == len(axes), f'got duplicate axes, {axes}'
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value).to(x.dtype)
    # axes = [axes if axes>=0 else axes+x.ndim for axes in axes]
    temp = [[0,0] for _ in range(x.ndim)]
    for i, ax in enumerate(axes):
        temp[ax][0] = pad[i]
        temp[ax][1] = pad[i+len(axes)]
    pad = [a for l in temp[::-1] for a in l]
    return torch.nn.functional.pad(x, pad, mode=mode, value=value)

def add_pad_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    if state.graph.opset <11:
        assert len(node.inputs) ==1 , f'{node.name} with operator {node.op} should have 1 input in opset (1 to 10), but got {len(node.inputs)}'
        mode = node.attrs.get('mode','constant')
        paddings = node.attrs.get('paddings')
        value = node.attrs.get('value',0.0)
        inp = utils.get_input_from_node(node.inputs[0], torch_graph,torch_nodes, torch_module, torch.nn.Parameter if inp.shape else torch.Tensor)
        torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.pad, (inp,paddings, value), dict(mode=mode), name=node.name)
    elif state.graph.opset <18:
        assert len(node.inputs) ==3 , f'{node.name} with operator {node.op} should have 3 inputs in opset (11 to 17), but got {len(node.inputs)}'
        types = [torch.nn.Parameter, list, list]
        args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
        mode = node.attrs.get('mode','constant')
        torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.pad, tuple(args), dict(mode=mode), name=node.name)
    else:
        assert len(node.inputs) ==4 , f'{node.name} with operator {node.op} should have 4 inputs in opset (18 to 17), but got {len(node.inputs)}'
        types = [torch.nn.Parameter, list, list, list]
        args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
        mode = node.attrs.get('mode','constant')
        torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.pad, tuple(args), dict(mode=mode), name=node.name) 