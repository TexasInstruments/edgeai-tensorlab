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
from operator import getitem

def torch_slice(x, starts, ends, axes, steps=None):
    if not isinstance(axes, (list, tuple, torch.Tensor)):
        axes = [axes]
    if not isinstance(starts, (list, tuple, torch.Tensor)):
        starts = [starts]
    if not isinstance(ends, (list, tuple, torch.Tensor)):
        ends = [ends]
    if not isinstance(axes, torch.Tensor):
        axes = torch.tensor(axes)
    if steps is None:
        steps = torch.ones_like(axes.detach().clone())
    temp = None
    if not  isinstance(x, torch.Tensor):
        temp = x
        x = torch.tensor(x)
    dim = x.dim() if isinstance(x, torch.Tensor) else len(x)
    slices = [[None, None, None] for _ in range(dim)]
    x = temp or x
    for i, axis in enumerate(axes.tolist()):
        slices[axis][0] = starts[i]
        slices[axis][1] = ends[i]
        slices[axis][2] = steps[i]
    if len(slices)==1:
        return getitem(x, slice(*slices[0]))
    for i, slc in enumerate(slices):
        if slc[0] == None:
            continue
        slc_ = [slice(None, None, None) for _ in range(dim)]
        slc_[i] = slice(*slc)
        x = getitem(x, tuple(slc_)) if isinstance(x, torch.Tensor) else getitem(x, slc_[0])
    return x

def add_slice_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 1<=len(node.inputs)<=5, f'{node.name} with operator {node.op} should have 1 or 5 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, list, list, list, list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    kwargs = dict()
    if len(args) == 3:
        args.append(0)
    if 'starts' in node.attrs:
        starts = node.attrs.get('starts')
        kwargs['starts'] = starts
    if 'ends' in node.attrs:
        ends = node.attrs.get('ends')
        kwargs['ends'] = ends
    if 'axes' in node.attrs:
        axes = node.attrs.get('axes')
        kwargs['axes'] = axes
    if 'steps' in node.attrs:
        steps = node.attrs.get('steps')
        kwargs['steps'] = steps
    torch_nodes[node.name] = torch_graph.call_function(torch_slice, tuple(args),  kwargs, name=node.name)
