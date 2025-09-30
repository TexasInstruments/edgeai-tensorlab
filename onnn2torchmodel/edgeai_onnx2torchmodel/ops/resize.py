import torch
import onnx_graphsurgeon as gs
from . import utils

# TODO proper support for all args and kwargs
def torch_resize(x, roi=None, scales=None, sizes=None, coordinate_transformation_mode='half_pixel', cubic_coeff_a=-0.75, exclude_outside=False, extrapolation_value=0, mode='nearest', nearest_mode='round_prefer_floor', antialias=False):
    kwargs = dict(
        antialias=antialias
    )
    if mode in ('linear','bilinear', 'bicubic', 'trilinear'):
        kwargs.update(dict(align_corners=coordinate_transformation_mode == 'align_corners'))
    scale_len = 0
    if mode == 'linear':
        scale_len = 1
    elif mode in ('bilinear', 'bicubic'):
        scale_len = 2
    elif mode == 'trilinear':
        scale_len = 3
    if scales :
        new_scales = []
        start= False
        for scale in scales:
            if not start and scale == 1:
                continue
            start= True
            new_scales.append(scale)
        scales = new_scales
        if scale_len and len(scales) < scale_len:
            scales = [1]*(scale_len-len(scales)) + scales
    if sizes :
        new_sizes = []
        start= False
        for i, size in enumerate(sizes):
            if not start and size == x.shape[i]:
                continue
            start= True
            new_sizes.append(size)
        sizes = new_sizes
        if scale_len and len(sizes) < scale_len:
            sizes = x.shape[-scale_len:-len(sizes)] + sizes
    
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