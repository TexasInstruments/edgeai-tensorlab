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

# TODO proper support for all args and kwargs
def torch_resize(x, roi=None, scales=None, sizes=None, coordinate_transformation_mode='half_pixel', cubic_coeff_a=-0.75, exclude_outside=False, extrapolation_value=0, mode='nearest', nearest_mode='round_prefer_floor', antialias=False):
    kwargs = dict(
        antialias=antialias
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
            
    if mode in ('linear','bilinear', 'bicubic', 'trilinear'):
        kwargs.update(dict(align_corners=coordinate_transformation_mode == 'align_corners'))
    
    return torch.nn.functional.interpolate(x, sizes, scales, mode=mode, **kwargs )

def add_resize_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph, torch_nodes:dict[str|torch.fx.Node], torch_module:torch.nn.Module):
    opset = state.graph.opset
    if opset < 11:
        assert len(node.inputs) == 2, f'{node.name} with operator {node.op} should have 2 input, but got {len(node.inputs)}'
        types = [torch.nn.Parameter, list]
    else:
        assert 1<= len(node.inputs) <= 4, f'{node.name} with operator {node.op} should have between 1 and 4 inputs, but got {len(node.inputs)}'
        types = [torch.nn.Parameter, torch.Tensor, list, list]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    kwargs = dict(node.attrs)
    if opset < 11:
        args = args[0:1] + [None] + args[1:]
    torch_nodes[node.name] = torch_graph.call_function(torch_resize, tuple(args),  kwargs, name=node.name)