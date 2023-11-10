# Copyright (c) 2018-2021, Texas Instruments
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

import os
import numpy as np
import torch
from edgeai_torchmodelopt import xnn
from edgeai_xvision import xvision


#################################################
def shape_as_string(shape=[]):
    shape_str = ''
    for dim in shape:
        shape_str += '_' + str(dim)
    return shape_str


def write_tensor_int(m=[], tensor=[], suffix='op', bitwidth=8, power2_scaling=True, file_format='bin',
                     rnd_type='rnd_sym', force_data_type=None,save_path=None):
    mn = tensor.min()
    mx = tensor.max()

    print(
        '{:6}, {:32}, {:10}, {:7.2f}, {:7.2f}'.format(suffix, m.name, m.__class__.__name__, tensor.min(), tensor.max()),
        end=" ")

    [tensor_scale, clamp_limits, tensor_signed] = xnn.utils.compute_tensor_scale(tensor, mn, mx, bitwidth, power2_scaling, force_data_type=force_data_type)
    #tensor_signed = min(mn, mx) < 0
    print("{:30} : {:15} : {:8.2f}".format(str(tensor.shape), str(tensor.dtype), tensor_scale), end=" ")

    print_weight_bias = False
    if rnd_type == 'rnd_sym':
        # use best rounding for offline quantities
        if suffix == 'weight' and print_weight_bias:
            no_idx = 0
            torch.set_printoptions(precision=32)
            print("tensor_scale: ", tensor_scale)
            print(tensor[no_idx])
        if tensor.dtype != torch.int64:
            tensor = xnn.utils.symmetric_round_tensor(tensor * tensor_scale)
        if suffix == 'weight' and print_weight_bias:
            print(tensor[no_idx])
    else:
        # for activation use HW friendly rounding
        if tensor.dtype != torch.int64:
            tensor = xnn.utils.upward_round_tensor(tensor * tensor_scale)
    tensor = tensor.clamp(clamp_limits[0], clamp_limits[1]).float()

    if bitwidth == 8 and tensor_signed:
        data_type = np.int8
        str_data_type = 'int8'
    elif bitwidth == 16 and tensor_signed:
        data_type = np.int16
        str_data_type = 'int16'
    elif bitwidth == 32 and tensor_signed:
        data_type = np.int32
        str_data_type = 'int32'
    elif bitwidth == 8 and not tensor_signed:
        data_type = np.uint8
        str_data_type = 'uint8'
    elif bitwidth == 16 and not tensor_signed:
        data_type = np.uint16
        str_data_type = 'uint16'
    elif bitwidth == 32 and not tensor_signed:
        data_type = np.uint32
        str_data_type = 'uint32'
    else:
        exit("Bit width other 8,16,32 not supported for writing layer level op")

    tensor = tensor.cpu().numpy().astype(data_type)

    print("{:7} : {:7d} : {:7d}".format(str(tensor.dtype), tensor.min(), tensor.max()))
    root = os.getcwd()
    tensor_dir = os.path.join(root, save_path, '{}_{}_{}_{}_scale_{:010.4f}'.format(m.name, m.__class__.__name__, suffix, str_data_type, tensor_scale))
    if not os.path.exists(tensor_dir):
        os.makedirs(tensor_dir)

    if file_format == 'bin':
        tensor_name = tensor_dir + "/{}_shape{}.bin".format(m.name, shape_as_string(shape=tensor.shape))
        tensor.tofile(tensor_name)
    elif file_format == 'npy':
        tensor_name = tensor_dir + "/{}_shape{}.npy".format(m.name, shape_as_string(shape=tensor.shape))
        np.save(tensor_name, tensor)

    # utils_hist.comp_hist_tensor3d(x=tensor, name=m.name, en=True, dir=m.name, log=True, ch_dim=0)
    return tensor_scale


def write_tensor_float(m=[], tensor=[], suffix='op',save_path=None):
    mn = tensor.min()
    mx = tensor.max()

    print(
        '{:6}, {:32}, {:10}, {:7.2f}, {:7.2f}'.format(suffix, m.name, m.__class__.__name__, tensor.min(), tensor.max()))
    root = os.getcwd()
    tensor_dir = root + '/checkpoints/debug/test_vecs/' + '{}_{}_{}'.format(m.name, m.__class__.__name__, suffix)

    if not os.path.exists(tensor_dir):
        os.makedirs(tensor_dir)

    tensor_name = tensor_dir + "/{}_shape{}.npy".format(m.name, shape_as_string(shape=tensor.shape))
    np.save(tensor_name, tensor.data)


def write_tensor(data_type='int', m=[], tensor=[], suffix='op', bitwidth=8, power2_scaling=True, file_format='bin',
                 rnd_type='rnd_sym', force_data_type=None,save_path=None):
    if data_type == 'int':
        tensor_scale = write_tensor_int(m=m, tensor=tensor, suffix=suffix, rnd_type=rnd_type, bitwidth=bitwidth,
            file_format=file_format,force_data_type=force_data_type,save_path=save_path)
    elif data_type == 'float':
        write_tensor_float(m=m, tensor=tensor, suffix=suffix,save_path=save_path)
    return tensor_scale

def write_tensor_hook_function(m, inp, out, save_path=None, file_format='bin'):

    # Output
    if isinstance(out, (torch.Tensor)):
        tensor_scale_op = write_tensor(m=m, tensor=out, suffix='op', rnd_type='rnd_up', file_format=file_format, save_path=save_path)

    # Input(s)
    if type(inp) is tuple:
        # if there are more than 1 inputs
        for index, sub_ip in enumerate(inp[0]):
            if isinstance(sub_ip, (torch.Tensor)):
                tensor_scale_ip = write_tensor(m=m, tensor=sub_ip, suffix='ip_{}'.format(index), rnd_type='rnd_up',
                             file_format=file_format, save_path=save_path)
    elif isinstance(inp, (torch.Tensor)):
        tensor_scale_ip = write_tensor(m=m, tensor=inp, suffix='ip', rnd_type='rnd_up', file_format=file_format, save_path=save_path)

    # weights
    if hasattr(m, 'weight'):
        if isinstance(m.weight, torch.Tensor):
            tensor_scale_wt = write_tensor(m=m, tensor=m.weight, suffix='weight', rnd_type='rnd_sym', file_format=file_format, save_path=save_path)

    # bias
    if hasattr(m, 'bias'):
        if m.bias is not None:
            tensor_scale_bias = write_tensor(m=m, tensor=m.bias, suffix='bias', rnd_type='rnd_sym', bitwidth=16,
                force_data_type = 'signed', file_format=file_format, save_path=save_path)
