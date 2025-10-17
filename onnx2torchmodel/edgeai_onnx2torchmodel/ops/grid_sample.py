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

def add_grid_sample_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    mode = node.attrs.get('mode','linear')
    padding_mode = node.attrs.get('padding_mode', 'zeros')
    align_corners = node.attrs.get('align_corners',0) == 1
    if mode in ('linear', 'cubic'):
        mode = 'bi'+mode
    kwargs = dict(
        mode=mode,
        align_corners=align_corners,
        padding_mode=padding_mode
    )
    torch_nodes[node.name] = torch_graph.call_function(torch.nn.functional.grid_sample, tuple(args),  kwargs, name=node.name)

def torch_upsample(x, sizes=None, scales=None, mode='nearest', align_corners=None):
    kwargs = dict(
    )
    scale_len = x.dim()-2
    if scales :
        new_scales = []
        start= False
        for scale in scales:
            if not start and scale == 1:
                continue
            start= True
            new_scales.append(scale)
        scales = new_scales
        if len(scales) == 0:
            scales = [1.0]
        if scale_len and len(scales) < scale_len:
            scales = [1.0]*(scale_len-len(scales)) + scales
        if len(scales) == 2:
            if mode in ('linear','cubic'):
                mode = 'bi' + mode 
            
        elif len(scales) == 3 and mode == 'linear':
            mode = 'trilinear'
        
    if sizes :
        new_sizes = []
        start= False
        for i, size in enumerate(sizes):
            if not start and size == x.shape[i]:
                continue
            start= True
            new_sizes.append(size)
        sizes = new_sizes
        if len(sizes) == 0:
            sizes = x.shape[-1:]
        if scale_len and len(sizes) < scale_len:
            sizes = x.shape[-scale_len:-len(sizes)] + tuple(sizes)
        if len(sizes) == 2 :
            if mode in ('linear','cubic'):
                mode = 'bi' + mode 
        elif len(sizes) == 3 and mode == 'linear':
            mode = 'trilinear'
    return torch.nn.functional.interpolate(x, sizes, scale, mode, align_corners)
    

def add_upsample_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter, list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    mode = node.attrs.get('mode','nearest')
    kwargs = dict(
        mode=mode
    )
    if len(args) == 2:
        args = args[0],None,args[1] 
    else:
        args.append(None)
    if 'scales' in node.attrs:
        args.append(node.attrs.get('scales'))
    elif any(x in node.attrs for x in ('height_scale','width_scale')):
        kwargs['height_scale'] = node.attrs.get('height_scale',1)
        kwargs['width_scale'] = node.attrs.get('width_scale',1)
        args .append((kwargs['height_scale'],kwargs['width_scale']))
    
    torch_nodes[node.name] = torch_graph.call_function(torch_upsample, tuple(args),  kwargs, name=node.name)