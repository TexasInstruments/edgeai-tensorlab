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

def torch_depth_to_space(x, block_size, mode='DCR'):
    c,h,w = x.shape[-3:]
    if mode == 'DCR':
        x = x.reshape( list(x.shape[:-3]) +[block_size, block_size, c//(block_size**2), h, w])
        perm = list(range(len(x.shape)))
        perm[-5:] = perm[-3], perm[-2], perm[-5], perm[-1], perm[-4]
        x = x.permute(perm)
    elif mode == 'CRD':
        x = x.reshape( list(x.shape[:-3]) +[c//(block_size**2), block_size, block_size, h, w])
        perm = list(range(len(x.shape)))
        perm[-4:] = perm[-2], perm[-4], perm[-1], perm[-3]
        x = x.permute(perm)
    else :
        raise ValueError(f'Unsupported mode {mode}')
    return x.reshape(list(x.shape[:-4]) +[h*block_size, w*block_size])

def add_depth_to_space_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 1, f'{node.name} with operator {node.op} should have 1 input, but got {len(node.inputs)}'
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    block_size = node.attrs.get('block_size')
    mode = node.attrs.get('mode','DCR')
    torch_nodes[node.name] = torch_graph.call_function(torch_depth_to_space, tuple(args),  dict(block_size=block_size, mode=mode), name=node.name)
    