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

def add_clip_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [ torch.Tensor, list, list]
    assert 1<=len(node.inputs)<=3, f'{node.name} with operator {node.op} should have between 1 and 3 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    kwargs = {}
    if 'min' in node.attrs:
        kwargs['min'] = node.attrs['min']
    if 'max' in node.attrs:
        kwargs['max'] = node.attrs['max']
    if state.module_based:
        module = utils.WrappedModule(node.op, torch_module, torch.clip, args, kwargs)
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch.clip, tuple(args), kwargs, name=node.name)
    for attr in node.attrs:
        if attr in kwargs:
            continue
        torch_nodes[node.name].meta[attr] = node.attrs[attr]