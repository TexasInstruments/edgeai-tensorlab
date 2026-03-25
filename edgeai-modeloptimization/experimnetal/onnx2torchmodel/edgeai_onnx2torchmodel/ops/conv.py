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
import warnings
import onnx_graphsurgeon as gs
from . import utils
from .pad import Pad

def get_torch_conv_module(node:gs.Node, torch_module:torch.nn.Module):
    groups = node.attrs.get('group', 1)
    out, inc = node.inputs[1].shape[:2]
    inc = inc*groups
    kernel_size = node.attrs.get('kernel_shape', node.inputs[1].shape[2:])
    stride = node.attrs.get('strides', 1)
    padding = node.attrs.get('pads', 0)
    dilation = node.attrs.get('dilations', 1)
    auto_pad = node.attrs.get('auto_pad', 'NOTSET')
    # TODO auto_pad cases install
    if auto_pad == 'SAME_UPPER':
        pass
    elif auto_pad == 'SAME_LOWER':
        pass
    elif auto_pad == 'VALID':
        pass
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
    has_bias = len(node.inputs) == 3
    if len(kernel_size)==1:
        cls = torch.nn.Conv1d
        bn_cls = torch.nn.BatchNorm1d
    elif len(kernel_size)==2:
        cls = torch.nn.Conv2d
        bn_cls = torch.nn.BatchNorm2d
    elif len(kernel_size)==3:
        cls = torch.nn.Conv3d
        bn_cls = torch.nn.BatchNorm3d
    else:
        raise NotImplementedError('conv only supports 1d, 2d and 3d inputs but got {}D'.format(len(kernel_size)))
    module = cls(inc, out, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=has_bias)
    module.weight = getattr(torch_module, node.inputs[1].name)
    if has_bias:
        module.bias = getattr(torch_module, node.inputs[2].name)
    if add_padding and padding:
        module = torch.nn.Sequential(Pad(old_padding), module)
    else:
        module = torch.nn.Sequential(module)

    if torch_module.training:
        module.append(
            bn_cls(out, eps=1e-5, momentum=0.1),
        )
        module[-1].running_var *= 1-(1e-5)
            
    module.training = torch_module.training
    torch_module.add_module(node.name, module)
    return module

def add_conv_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 3>=len(node.inputs) >= 2, f'{node.name} with operator {node.op} should have between 2 and 3 inputs, but got {len(node.inputs)}'
    changed = False
    if len(node.outputs[0].outputs) == 1 and node.outputs[0].outputs[0].op == 'BatchNormalization':
        state.training = False
        torch_module.training = False
        changed = True
    types = [torch.nn.Parameter  for inp in node.inputs]
    args = [utils.get_input_from_node(node, inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    node_name = node.name+'_'+node.op        
    if state.training  and not  all(isinstance(t,c) for t,c in zip(node.inputs,(gs.Variable, gs.Constant, gs.Constant))):
        warnings.warn(f'{node_name} with operator {node.op} is not suitable for conversion with training mode changing to inference mode. this operator should only have variable input, constant weight and bias (if any) for training.')
        state.training = False
        torch_module.training = False
        changed = True
    if all(isinstance(t,c) for t,c in zip(node.inputs,(gs.Variable, gs.Constant, gs.Constant))):
        module = get_torch_conv_module(node, torch_module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args[0:1]),)
    else:
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
        if add_padding and padding:
            pad_module = Pad(old_padding)
            torch_module.add_module(node.name+'_pad', pad_module)
            padding_node = torch_graph.call_module(node.name+'_pad', tuple(args[0:1]))
            args[0] = padding_node
        if state.module_based:
            module = utils.WrappedModule(node.name, node.op, torch_module, func, args, kwargs)
            torch_module.add_module(node.name, module)
            args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
            torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
        else:
            torch_nodes[node.name] = torch_graph.call_function(func, tuple(args),  kwargs, name=node.name)
    if changed:
        state.training = True
        torch_module.training = True

def add_conv_integer_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    raise NotImplementedError(f"{node.name} with operator {node.op} is not implemented")

def get_torch_conv_transpose_module(node:gs.Node, torch_module:torch.nn.Module):
    groups = node.attrs.get('group', 1)
    out, inc = node.inputs[1].shape[:2]
    inc = inc*groups
    kernel_size = node.attrs.get('kernel_shape', node.inputs[1].shape[2:])
    stride = node.attrs.get('strides', 1)
    padding = node.attrs.get('pads', 0)
    dilation = node.attrs.get('dilations', 1)
    output_padding = node.attrs.get('output_padding',0)
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
        stride = stride,
        padding = padding,
        dilation = dilation,
        groups = groups,
        output_padding = output_padding,
    )
    has_bias = len(node.inputs) == 3
    if len(kernel_size)==1:
        cls = torch.nn.ConvTranspose1d
        # bn_cls = torch.nn.BatchNorm1d
    elif len(kernel_size)==2:
        cls = torch.nn.ConvTranspose2d
        # bn_cls = torch.nn.BatchNorm2d
    elif len(kernel_size)==3:
        cls = torch.nn.ConvTranspose3d
        # bn_cls = torch.nn.BatchNorm3d
    else:
        raise NotImplementedError('conv only supports 1d, 2d and 3d inputs but got {}D'.format(len(kernel_size)))
    module = cls(inc, out, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=has_bias)
    module.weight = getattr(torch_module, node.inputs[1].name)
    if has_bias:
        module.bias = getattr(torch_module, node.inputs[2].name)
    if add_padding and padding:
        module = torch.nn.Sequential(Pad(old_padding), module)
    else:
        module = torch.nn.Sequential(module)
            
    module.training = torch_module.training
    torch_module.add_module(node.name, module)
    return module

def add_conv_transpose_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert 3>=len(node.inputs) >= 2, f'{node.name} with operator {node.op} should have between 2 and 3 inputs, but got {len(node.inputs)}'
    types = [torch.nn.Parameter  for inp in node.inputs]
    args = [utils.get_input_from_node(node, inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
    if all(isinstance(t,c) for t,c in zip(node.inputs,(gs.Variable, gs.Constant, gs.Constant))):
        module = get_torch_conv_transpose_module(node, torch_module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args[0:1]),)
    else:
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
        if add_padding and padding:
            pad_module = Pad(old_padding)
            torch_module.add_module(node.name+'_pad', pad_module)
            padding_node = torch_graph.call_module(node.name+'_pad', tuple(args[0:1]))
            args[0] = padding_node
        if state.module_based:
            module = utils.WrappedModule(node.name, node.op, torch_module, func, args, kwargs)
            torch_module.add_module(node.name, module)
            args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
            torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
        else:
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
    args = [utils.get_input_from_node(node, inp, torch_graph,torch_nodes, torch_module,t) for inp,t in zip(node.inputs, types)]
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
    
    if state.module_based:
        module = utils.WrappedModule(node.name, node.op, torch_module, torch_deform_conv, args, kwargs)
        torch_module.add_module(node.name, module)
        args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_deform_conv, tuple(args),  kwargs, name=node.name)
        