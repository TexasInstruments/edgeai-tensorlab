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

torch.library.custom_op.slice = None
@torch.library.custom_op('custom_ops::slice', mutates_args=())
def single_slice(x:torch.Tensor, axis:int, start:int, end:int,  step:int)-> torch.Tensor:
    # return torch.ops.aten.slice.Tensor(x.clone(),axis, start, end, step)
    dim = x.dim() if isinstance(x, torch.Tensor) else len(x)
    slices = [slice(None, None, None) for _ in range(min(axis+1,dim))]
    slices[axis] = slice(start ,end, step)
    return getitem(x.clone(), tuple(slices))

@single_slice.register_fake
def _(x, axis, start, end, step):
    return torch.ops.aten.slice.Tensor(x,axis, start, end, step)

func = torch.library.custom_op.slice or single_slice

def adjust_for_slice(x):
    if not isinstance(x, (list, tuple, torch.Tensor)):
        x = [x]
    if  not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if x.dim() == 0:
        x = x.unsqueeze(0)
    return x
    

def torch_slice(x, starts, ends, axes=None, steps=None):
    dim = x.dim() if isinstance(x, torch.Tensor) else len(x)
    starts = adjust_for_slice(starts).cpu().tolist()
    ends = adjust_for_slice(ends).cpu().tolist()
    axes = axes or list(range(len(starts)))
    axes = adjust_for_slice(axes) 
    if steps is None:
        steps = torch.ones_like(axes.detach().clone())
    steps = adjust_for_slice(steps).cpu().tolist()
    axes = torch.where(axes<0, axes+dim, axes)
    axes = axes.cpu().tolist()
    
    temp = None
    if not  isinstance(x, torch.Tensor):
        temp = x
        x = torch.tensor(x)
    
    for start, end, axis, step in zip(starts, ends, axes, steps):
        # d = x.shape[]
        x = func(x, axis, start, end, step)
    
    return x if temp is None else x.cpu().tolist()

def add_slice_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 1<=len(node.inputs)<=5, f'{node.name} with operator {node.op} should have 1 or 5 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, list, list, list, list]
    args = [utils.get_input_from_node(node, inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    kwargs = dict()
    if 'starts' in node.attrs:
        starts = node.attrs.get('starts')
        kwargs['starts'] = starts
    if 'ends' in node.attrs:
        ends = node.attrs.get('ends')
        kwargs['ends'] = ends
    if 'axes' in node.attrs:
        axes = node.attrs.get('axes')
        kwargs['axes'] = axes

    for i in range(1, len(args)):
        arg = args[i]
        if isinstance(arg, torch.fx.Node) and arg.op == 'get_attr':
            args[i] = arg = getattr(torch_module, arg.target)
            if isinstance(arg, torch.Tensor):
                args[i] = arg.cpu().tolist()
    if state.module_based:
        module = utils.WrappedModule(node.name, node.op, torch_module, torch_slice, args, kwargs,)
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_slice, tuple(args),  kwargs, name=node.name)
