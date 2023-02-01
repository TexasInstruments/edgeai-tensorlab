#################################################################################
# Copyright (c) 2018-2021, Texas Instruments Incorporated - http://www.ti.com
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
from . import module_utils

def forward_get_complexity(model, inp):
    num_flops = forward_count_flops(model, inp)
    num_params = count_params(model)
    return num_flops, num_params


def forward_count_flops(model, inp):
    _add_hook(model, _count_flops_func)
    _ = model(inp)
    num_flops = 0
    for m in model.modules():
        num_flops += m.__num_flops__
    #
    _remove_hook(model)
    return num_flops

def fw_bw_count_flops(model, inp):
    _add_hook_fw_bw(model, _count_fw_bw_flops_func)
    _ = model(inp)
    num_fw_bw_flops = 0
    op_vol = 0
    for m in model.modules():
        num_fw_bw_flops += m.__num_fw_bw_flops__
        op_vol += m.__op_vol__
    #
    _remove_hook_fw_bw(model)
    return num_fw_bw_flops, op_vol


def count_params(model):
    layer_params = [p.numel() for p in model.parameters() if p.requires_grad]
    num_params = sum(layer_params)
    return num_params


def _count_flops_func(m, inp, out):
    # trained calibration/quantization can do model surgery and return extra outputs - ignroe them
    if isinstance(out, (list,tuple)):
        out = out[0]
    #
    if module_utils.is_conv_deconv(m):
        num_pixels = (out.shape[2] * out.shape[3])
        # Note: channels_in taken from weight shape is already divided by m.groups - no need to divide again
        channels_out, channels_in, kernel_height, kernel_width = m.weight.shape
        macs_per_pixel = (channels_out * channels_in *  kernel_height * kernel_width)
        num_flops = 2 * macs_per_pixel * num_pixels
        if hasattr(m, 'bias') and (m.bias is not None):
            num_flops += m.weight.shape[0]
        #
        m.__num_flops__ = num_flops
    else:
        m.__num_flops__ = 0
    #
def _count_fw_bw_flops_func(m, inp, out):
    # trained calibration/quantization can do model surgery and return extra outputs - ignroe them
    if isinstance(out, (list,tuple)):
        out = out[0]
    #
    if module_utils.is_conv_deconv(m):
        num_pixels = (out.shape[2] * out.shape[3])
        # Note: channels_in taken from weight shape is already divided by m.groups - no need to divide again
        channels_out, channels_in, kernel_height, kernel_width = m.weight.shape
        macs_per_pixel = (channels_out * channels_in *  kernel_height * kernel_width)
        num_flops = 2 * macs_per_pixel * num_pixels
        if hasattr(m, 'bias') and (m.bias is not None):
            num_flops += m.weight.shape[0]
        #1 for fw + 2 for bw
        m.__num_fw_bw_flops__ = num_flops*3
        m.__op_vol__ = channels_out*(out.shape[2] * out.shape[3])
    elif module_utils.is_bn(m):
        channels_out = m.weight.shape[0]
        out_vol = channels_out*(out.shape[2] * out.shape[3])
        num_flops = 2 * out_vol
        # 1 for fw + 2 for bw + 1 for recomputing BN to save memory
        m.__num_fw_bw_flops__ = num_flops*4
        m.__op_vol__ = 0

def _add_hook(module, hook_func):
    for m in module.modules():
        m.__count_flops_hook__ = m.register_forward_hook(hook_func)
        m.__num_flops__ = 0


def _add_hook_fw_bw(module, hook_func):
    for m in module.modules():
        m.__count_fw_bw_flops_hook__ = m.register_forward_hook(hook_func)
        m.__num_fw_bw_flops__ = 0
        m.__op_vol__ = 0


def _remove_hook(module):
    for m in module.modules():
        if hasattr(m, '__count_flops_hook__'):
            m.__count_flops_hook__.remove()
        #
        if hasattr(m, '__num_flops__'):
            del m.__num_flops__
        #

def _remove_hook_fw_bw(module):
    for m in module.modules():
        if hasattr(m, '__count_fw_bw_flops_hook__'):
            m.__count_fw_bw_flops_hook__.remove()
        #
        if hasattr(m, '__num_fw_bw_flops__'):
            del m.__num_fw_bw_flops__

