#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################
import torch


# straight-through estimation (STE) - this is the preferred mode for propagating outputs for quantization
# the backward gradients uses x (i.e. the input itself) and not y (the output)
# because the effect of y is completely detached during forward
def propagate_quant_ste(x, y):
    # this works functionally as STE, but exports an onnx graph containing
    # all the operators used to compute y as well
    # out = x + (y - x).detach()
    #
    # this is another way of doing STE. in this case the operators used to generate y are skipped from onnx graph
    out = x.clone()
    out.data = y.data
    return out


# quantized through estimation - typically not used.
def propagate_quant_qte(x, y):
    return y


def round_func(x):
    y = torch.round(x)
    return y


def round_g(x):
    return propagate_quant_ste(x, round_func(x))


def round_sym_func(x):
    rnd = (-0.5)*(x<0).float() + (0.5)*(x>=0).float()
    y = (x+rnd).int().float()
    return y


def round_sym_g(x):
    return propagate_quant_ste(x, round_sym_func(x))


def round_up_func(x):
    y = torch.floor(x+0.5)
    return y


def round_up_g(x):
    return propagate_quant_ste(x, round_up_func(x))


def round2_func(x):
    y = torch.pow(2,torch.round(torch.log2(x)))
    return y


def round2_g(x):
    return propagate_quant_ste(x, round2_func(x))


def ceil_func(x):
    y = torch.ceil(x)
    return y


def ceil_g(x):
    return propagate_quant_ste(x, ceil_func(x))


def ceil2_func(x):
    y = torch.pow(2,torch.ceil(torch.log2(x)))
    return y


def ceil2_g(x):
    return propagate_quant_ste(x, ceil2_func(x))


def floor2_func(x):
    y = torch.pow(2,torch.floor(torch.log2(x)))
    return y


def floor2_g(x):
    return propagate_quant_ste(x, floor2_func(x))


def quantize_dequantize_func(x, scale_tensor, width_min:float, width_max:float, power2:bool, axis:int, round_type:str='round_up'):
    # clip values need ceil2 and scale values need floor2
    scale_tensor = floor2_func(scale_tensor) if power2 else scale_tensor
    x_scaled = (x * scale_tensor)

    # round
    if round_type == 'round_up':    # typically for activations
        rand_val = 0.5
        x_scaled_round = torch.floor(x_scaled+rand_val)
    elif round_type == 'round_sym': # typically for weights
        rand_val = (-0.5) * (x < 0).float() + (0.5) * (x >= 0).float()
        x_scaled_round = (x_scaled+rand_val).int().float()
    elif round_type == 'round_torch':
        x_scaled_round = torch.round(x_scaled)
    elif round_type is None:
        x_scaled_round = x_scaled
    else:
        assert False, 'quantize_dequantize_func: unknown round tyoe'
    #
    # invert the scale
    scale_inv = scale_tensor.pow(-1.0)
    # clamp
    x_clamp = torch.clamp(x_scaled_round, width_min, width_max)
    y = x_clamp * scale_inv
    return y, x_scaled_round


# quantization operation with STE gradients
def quantize_dequantize_g(x, *args, **kwargs):
    return propagate_quant_ste(x, quantize_dequantize_func(x, *args, **kwargs)[0])


# torch now natively supports quantization operations
def quantize_dequantize_torch_g(x, scale_tensor, width_min:float, width_max:float, power2:bool, axis:int, round_type:str='round_up'):
    # apply quantization
    if scale_tensor.dim()>0:
        device = x.device
        axis_size = int(x.size(axis))
        scale_tensor = scale_tensor.reshape(axis_size)
        zero_point = torch.zeros(axis_size).to(device=device, dtype=torch.long)
        y = torch.fake_quantize_per_channel_affine(x, scale=scale_tensor, zero_point=zero_point, axis=axis,
                quant_min=int(width_min), quant_max=int(width_max))
    else:
        y = torch.fake_quantize_per_tensor_affine(x, scale=float(scale_tensor), zero_point=0,
                quant_min=int(width_min), quant_max=int(width_max))
    #
    return y


# a clamp function that can support tensor arguments for min and max
# becomes a simple clamp/clip during evaluation
def clamp_g(x, min, max, training, inplace=False, requires_grad=False):
    if x is None:
        return x
    #
    # in eval mode, torch.clamp can be used
    # the graph exported in eval mode will be simpler and have fixed constants that way.
    if training:
        if requires_grad:
            # torch's clamp doesn't currently work with min and max as tensors
            # TODO: replace with this, when torch clamp supports tensor arguments:
            # TODO:switch back to min/max if you want to lean the clip values by backprop
            zero_tensor = torch.zeros_like(x.view(-1)[0])
            min = zero_tensor + min
            max = zero_tensor + max
            y = torch.max(torch.min(x, max), min)
        else:
            # clamp takes less memory - using it for now
            y = torch.clamp_(x, min, max) if inplace else torch.clamp(x, min, max)
        #
    else:
        # use the params as constants for easy representation in onnx graph
        y = torch.clamp_(x, float(min), float(max)) if inplace else torch.clamp(x, float(min), float(max))
    #
    return y


###################################################
# from torchvision shufflenetv2
# https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


###################################################
def channel_split_by_chunks(x, chunks):
    branches = torch.chunk(x,chunks,dim=1)
    return branches


def channel_split_by_size(x, split_size):
    branches = torch.split(x,split_size,dim=1)
    return branches


def channel_split_by_index(x, split_index):
    split_list = [split_index, int(x.size(1))-split_index]
    branches = torch.split(x,split_list,dim=1)
    return branches


def channel_slice_by_index(x, split_index):
    return (x[:,:split_index,...],x[:,split_index:,...])


###################################################
def crop_grid(features, crop_offsets, crop_size):
    b = int(features.size(0))
    h = int(features.size(2))
    w = int(features.size(3))
    zero = features.new(b, 1).zero_()
    one = zero + 1.0
    y_offset = crop_offsets[:,0] / (h-1)
    x_offset = crop_offsets[:,1] / (w-1)
    theta = torch.cat([one, zero, x_offset, zero, one, y_offset], dim=1).view(-1,2,3)
    grid = torch.nn.functional.affine_gird(theta, (b, 1, crop_size[0], crop_size[1]))
    out = torch.nn.functional.grid_sample(features, grid)
    return out


###################################################
def split_output_channels(output, output_channels):
    if isinstance(output, (list, tuple)):
        return output
    elif len(output_channels) == 1:
        return [output]
    else:
        start_ch = 0
        task_outputs = []
        for num_ch in output_channels:
            if len(output.shape) == 3:
                task_outputs.append(output[start_ch:(start_ch + num_ch), ...])
            elif len(output.shape) == 4:
                task_outputs.append(output[:, start_ch:(start_ch + num_ch), ...])
            else:
                assert False, 'incorrect dimensions'
            # --
            start_ch += num_ch
        # --
        return task_outputs
