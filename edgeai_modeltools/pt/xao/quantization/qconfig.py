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
import torch.ao.quantization as quantization
from . import observer


def _get_qat_qconfig(backend=None, weight_observer=None, activation_observer=None,
                     weight_qscheme=None, activation_qscheme=None):
    # we support only symmetric types (per tensor or per channel) for weight
    assert weight_qscheme in (torch.per_tensor_symmetric, torch.per_channel_symmetric), 'weight_qscheme must be one of the symmetric types'
    weight_dtype = torch.qint8
    weight_quant_min = -127
    weight_quant_max = 127

    # we can support both symmetric and affine types for activation
    if activation_qscheme in (torch.per_tensor_symmetric, torch.per_channel_symmetric):
        activation_dtype = torch.qint8
        activation_quant_min = -127
        activation_quant_max = 127
    else:
        activation_dtype = torch.quint8
        activation_quant_min = 0
        activation_quant_max = 255
    #
    activation_reduce_range = (backend == 'fbgemm') and (activation_qscheme != torch.per_tensor_symmetric)

    weight_config = quantization.FusedMovingAvgObsFakeQuantize.with_args(
        observer=weight_observer,
        quant_min=weight_quant_min, quant_max=weight_quant_max,
        dtype=weight_dtype,
        qscheme=weight_qscheme)

    activation_config = quantization.FusedMovingAvgObsFakeQuantize.with_args(
        observer=activation_observer,
        quant_min=activation_quant_min, quant_max=activation_quant_max,
        reduce_range=activation_reduce_range,
        dtype=activation_dtype, qscheme=activation_qscheme)

    qconfig = quantization.QConfig(activation=activation_config, weight=weight_config)
    return qconfig


def get_per_tensor_symmetric_power2_qat_qconfig(backend=None):
    return _get_qat_qconfig(backend=backend,
                            weight_observer=observer.MovingAverageMinMaxObserverPower2,
                            activation_observer=observer.MovingAverageMinMaxObserverPower2,
                            weight_qscheme=torch.per_tensor_symmetric,
                            activation_qscheme=torch.per_tensor_symmetric)

get_basic_qat_qconfig = get_per_tensor_symmetric_power2_qat_qconfig


def get_per_channel_affine_qat_qconfig(backend=None):
    return _get_qat_qconfig(backend=backend,
                            weight_observer=quantization.MovingAveragePerChannelMinMaxObserver,
                            activation_observer=quantization.HistogramObserver,
                            weight_qscheme=torch.per_channel_symmetric,
                            activation_qscheme=torch.per_tensor_affine)

get_advanced_qat_qconfig = get_per_channel_affine_qat_qconfig


# can also use torch.ao.quantization.get_default_qat_qconfig


def get_qat_qconfig_for_target_device(backend, target_device='TDA4VM'):
    ''''this is initial implementation. we can implement more target_device specific qconfigs later'''
    if target_device is None:
        return get_basic_qat_qconfig(backend=backend) # pytorch default qconfig
    elif target_device.lower() == 'tda4vm':
        return get_basic_qat_qconfig(backend=backend)
    else:
        return get_advanced_qat_qconfig(backend=backend)
