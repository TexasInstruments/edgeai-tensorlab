import torch
import onnx_graphsurgeon as gs
from . import utils


def add_conv_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 3>=len(node.inputs) >= 2, f'{node.name} with operator {node.op} should have between 2 and 3 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter  for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    kernel_size = node.attrs.get('kernel_shape')
    stride = node.attrs.get('strides')
    padding = node.attrs.get('pads')
    dilation = node.attrs.get('dilations')
    groups = node.attrs.get('group', 1)
    auto_pad = node.attrs.get('auto_pad', 'NOTSET')
    # TODO auto_pad cases install
    if auto_pad == 'SAME_UPPER':
        pass
    elif auto_pad == 'SAME_LOWER':
        pass
    elif auto_pad == 'VALID':
        pass
    
    padding = [padding[0], padding[2]]
    
    kwargs = dict(
        stride = stride,
        padding = padding,
        dilation = dilation,
        groups = groups,
    )
    
    if len(kernel_size) == 1:
        func = torch.nn.functional.conv1d
    elif len(kernel_size) == 2:
        func = torch.nn.functional.conv2d
    elif len(kernel_size) == 3:
        func = torch.nn.functional.conv3d
    else:
        raise ValueError(f'node {node.name} has no kernel_size')
    torch_nodes[node.name] = torch_graph.call_function(func, tuple(args),  kwargs, name=node.name)

def add_conv_integer_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")

def add_conv_transpose_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 3>=len(node.inputs) >= 2, f'{node.name} with operator {node.op} should have between 2 and 3 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter  for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    kernel_size = node.attrs.get('kernel_shape')
    stride = node.attrs.get('strides')
    padding = node.attrs.get('pads')
    output_padding = node.attrs.get('output_padding')
    dilation = node.attrs.get('dilations')
    groups = node.attrs.get('group',1)
    auto_pad = node.attrs.get('auto_pad', 'NOTSET')
    # TODO add support for output_shape
    output_shape = node.attrs.get('output_shape')
    # TODO auto_pad cases install
    if auto_pad == 'SAME_UPPER':
        pass
    elif auto_pad == 'SAME_LOWER':
        pass
    elif auto_pad == 'VALID':
        pass
    
    padding = [padding[0], padding[2]]
    kwargs = dict(
        stride = stride,
        padding = padding,
        dilation = dilation,
        groups = groups,
        output_padding = output_padding,
    )
    
    if len(kernel_size) == 1:
        func = torch.nn.functional.conv_transpose1d
    elif len(kernel_size) == 2:
        func = torch.nn.functional.conv_transpose2d
    elif len(kernel_size) == 3:
        func = torch.nn.functional.conv_transpose3d
    else:
        raise ValueError(f'node {node.name} has no kernel_size')
    torch_nodes[node.name] = torch_graph.call_function(func, tuple(args),  kwargs, name=node.name)

# TODO add support for offset_group
def deform_conv2d (x, weight, offset, bias=None, mask=None, kernl_size=None, stride=1, padding=0, dilation=1, groups=1, offset_group=1):
    pad_h          = padding[0]
    pad_w          = padding[1]
    dilation_h     = dilation[0]
    dilation_w     = dilation[1]
    stride_h       = stride[0]
    stride_w       = stride[1]
    
    num_kernel_elem = kernl_size[0] * kernl_size[1]
    
    o1 = offset[:, 0:(2*num_kernel_elem):2, :, :].to(torch.float32)
    o2 = offset[:, 1:(2*num_kernel_elem):2, :, :].to(torch.float32)
    b, n, h, w   = x.shape
    if mask is None:
        mask = torch.ones(b, num_kernel_elem, h, w, device=x.device, dtype=x.dtype)


    """ Deformable Convolution using GridSample
    x (1,Ni,H,W) --------------------> Pad (1, Ni, H+2, W+2) -------------------------> GridSample (1,Ni,Fr*Fc*H,W) -> Mul (1,Ni,9*H,W) ->  Reshape (1,Ni*9*H,W) -> Conv1x1 (1,No,H,W)
                                                                                                ^                          ^                                              ^
    offset_x (1,Fr*Fc,H,W) -> Unsqueeze(-1) (1,Fr*Fc,H,W,1) --                               |                          |                                              |
                                                                |-> Concat (1,Fr*Fc,H,W,2) -> Reshape (1,Fr*Fc*H,W,2)      |                                              |
    offset_y (1,Fr*Fc,H,W) -> Unsqueeze(-1) (1,Fr*Fc,H,W,1) --                                                          |                                              |
                                                                                                                        |                                              |
    mask (1,Fr*Fc,H,W) ------------------------------------------------------------------> Reshape (1, 1, 9*H, W) -------                                              |
                                                                                                                                                                        |
    weight (No,Ni,Fr,Fc) --------------------------------------------------------------------------------------------------------  -----> Reshape (No,Ni*Fr*Fc,1,1)-----
    """

    # 1. input feature padding
    x = torch.nn.functional.pad(x, [pad_w, pad_w, pad_h, pad_h, 0, 0])

    # 2. Feature map and mask size, where m = Fr*Fc
    b, n, h, w   = x.shape
    _, m, ho, wo = mask.shape
    _, _, fr, fc = weight.shape

    # 3. zero-offset location
    grid_y, grid_x = torch.meshgrid(
        torch.arange((dilation_h * (fr - 1)) // 2 + 0.5,
                        (dilation_h * (fr - 1)) // 2 + 0.5 + (ho - 1) * stride_h + 1,
                        1, device=o1.device, dtype=o1.dtype),
        torch.arange((dilation_w * (fc - 1)) // 2 + 0.5,
                        (dilation_w * (fc - 1)) // 2 + 0.5 + (wo - 1) * stride_w + 1,
                        1, device=o1.device, dtype=o1.dtype))

    grid_y = grid_y.repeat(m, 1, 1)
    grid_x = grid_x.repeat(m, 1, 1)

    # 4. 3x3 filter location without DCN
    k_y, k_x = torch.meshgrid(
        torch.arange(-(dilation_h*(fr-1))//2, (dilation_h*(fr-1))//2 +1, 1, device=o1.device, dtype=o1.dtype),
        torch.arange(-(dilation_w*(fc-1))//2, (dilation_w*(fc-1))//2 +1, 1, device=o1.device, dtype=o1.dtype))

    k_y = k_y.reshape(m, 1, 1)
    k_x = k_x.reshape(m, 1, 1)

    grid_y = grid_y + k_y
    grid_x = grid_x + k_x

    # 5. Normalizing sampling location (o2/o1 is x/y) 
    grid_y = (o1 + grid_y) / float(h) # 1x9xHxW # quantization does not su[pport double, making it quantization friendly
    grid_x = (o2 + grid_x) / float(w) # 1x9xHxW

    # in (x, y) order
    offset_grid = torch.cat((grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)), dim=-1) # 1x9xHxWx2

    # 6. Scale sampling location to [-1 to 1]
    offset_grid = float(2) * offset_grid - float(1)
    offset_grid = offset_grid.reshape(b, m*ho, wo, 2) # 1x(9*H)xWx2

    # 7. Sample features
    # x: 1xCx(H+2)x(W+2), offset_grid: 1x(9*H)xWx2
    # output: 1xCx(9*Ho)xWo
    sampling_feature = torch.nn.functional.grid_sample(x,
                                        offset_grid,
                                        mode='bilinear',
                                        padding_mode='zeros',
                                        align_corners=False)#.data

    sampling_feature = sampling_feature * mask.reshape(b, 1, m*ho, wo)
    sampling_feature = sampling_feature.reshape(b, n*m, ho, wo)

    # 8. Reshape self.weight to (1x1) weight
    weight_1x1 = weight.reshape(weight.shape[0], -1, 1, 1)

    # 9. 1x1 convolution
    out = torch.nn.functional.conv2d(sampling_feature, weight_1x1, bias=bias, groups=groups)
    return out

#TODO add support for 1d and 3d conv in deform_conv
def torch_deform_conv(x, weight, offset, bias=None, mask=None, kernl_size=None, stride=1, padding=0, dilation=1, groups=1, offset_group=1):
    if kernl_size is None:
        raise ValueError('kernl_size is None, it should be provided')
    if len(kernl_size) != 2 :   
        raise NotImplementedError('only support 2d deformable conv')
    else:
        return deform_conv2d(x, weight, offset, bias, mask, kernl_size, stride, padding, dilation, groups, offset_group)


def add_deform_conv_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 5>=len(node.inputs) >= 3, f'{node.name} with operator {node.op} should have between 3 and 5 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter  for inp in node.inputs]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    kernel_size = node.attrs.get('kernel_shape')
    stride = node.attrs.get('strides')
    padding = node.attrs.get('pads')
    dilation = node.attrs.get('dilations')
    groups = node.attrs.get('group', 1)
    offset_group = node.attrs.get('offset_group', 1)
    
    kwargs = dict(
        kernel_size = kernel_size,
        stride = stride,
        padding = padding,
        dilation = dilation,
        groups = groups,
        offset_group = offset_group
    )
    
    torch_nodes[node.name] = torch_graph.call_function(torch_deform_conv, tuple(args),  kwargs, name=node.name)
        