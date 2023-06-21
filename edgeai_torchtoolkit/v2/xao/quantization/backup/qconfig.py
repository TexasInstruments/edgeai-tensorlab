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
from . import qsettings


# the default qconfig
get_default_qat_qconfig = torch.ao.quantization.get_default_qat_qconfig


def _get_qat_qconfig(backend=None, weight_observer=None, activation_observer=None,
                     weight_qscheme=None, activation_qscheme=None):
    # we support only symmetric types (per tensor or per channel) for weight
    assert weight_qscheme in (torch.per_tensor_symmetric, torch.per_channel_symmetric), 'weight_qscheme must be one of the symmetric types'
    weight_dtype = torch.qint8
    weight_quant_min = qsettings.INT8_DTYPE_MIN_VALUE
    weight_quant_max = qsettings.INT8_DTYPE_MAX_VALUE

    if (activation_qscheme == torch.per_tensor_symmetric or activation_qscheme == torch.per_channel_symmetric) and \
        qsettings.USE_INT8_DTYPE_FOR_SYMMETRIC:
        activation_dtype = torch.qint8
        activation_quant_min = qsettings.INT8_DTYPE_MIN_VALUE
        activation_quant_max = qsettings.INT8_DTYPE_MAX_VALUE
    else:
        activation_dtype = torch.quint8
        activation_quant_min = qsettings.UINT8_DTYPE_MIN_VALUE
        activation_quant_max = qsettings.UINT8_DTYPE_MAX_VALUE
    #

    weight_fake_quant = quantization.FakeQuantize.with_args(
                                    observer=weight_observer,
                                    quant_min=weight_quant_min, quant_max=weight_quant_max,
                                    dtype=weight_dtype, qscheme=weight_qscheme, reduce_range=False)

    activation_fake_quant = quantization.FakeQuantize.with_args(
                                    observer=activation_observer,
                                    quant_min=activation_quant_min, quant_max=activation_quant_max,
                                    dtype=activation_dtype, qscheme=activation_qscheme, reduce_range=False)

    qconfig = quantization.QConfig(activation=activation_fake_quant, weight=weight_fake_quant)
    return qconfig


def get_per_tensor_symmetric_power2_qat_qconfig(backend=None, histogram_observer=qsettings.USE_HISTOGRAM_OBSERVER_DEFAULT):
    return _get_qat_qconfig(backend=backend,
                            weight_observer=(observer.HistogramObserverPower2 if histogram_observer else observer.MovingAverageMinMaxObserverPower2),
                            activation_observer=(observer.HistogramObserverPower2 if histogram_observer else observer.MovingAverageMinMaxObserverPower2),
                            weight_qscheme=torch.per_tensor_symmetric,
                            activation_qscheme=torch.per_tensor_symmetric)


get_basic_qat_qconfig = get_per_tensor_symmetric_power2_qat_qconfig


def get_per_tensor_affine_qat_qconfig(backend=None, histogram_observer=qsettings.USE_HISTOGRAM_OBSERVER_DEFAULT):
    return _get_qat_qconfig(backend=backend,
                            weight_observer=(quantization.HistogramObserver if histogram_observer else quantization.MovingAverageMinMaxObserver),
                            activation_observer=(quantization.HistogramObserver if histogram_observer else quantization.MovingAverageMinMaxObserver),
                            weight_qscheme=torch.per_tensor_symmetric,
                            activation_qscheme=torch.per_tensor_affine)


def get_per_channel_affine_qat_qconfig(backend=None, histogram_observer=qsettings.USE_HISTOGRAM_OBSERVER_DEFAULT):
    return _get_qat_qconfig(backend=backend,
                            weight_observer=quantization.MovingAveragePerChannelMinMaxObserver,
                            activation_observer=(quantization.HistogramObserver if histogram_observer else quantization.MovingAverageMinMaxObserver),
                            weight_qscheme=torch.per_channel_symmetric,
                            activation_qscheme=torch.per_tensor_affine)


def get_qat_qconfig_for_target_device(backend, target_device=None,
                                      histogram_observer=qsettings.USE_HISTOGRAM_OBSERVER_DEFAULT,
                                      symmetric_power2_quant=None, per_channel_weight_quant=False):
    ''''this is initial implementation. we can implement more target_device specific qconfigs later'''
    if target_device is None:
        # if target_device is not provided, we use the pytorch default qconfig
        # this is not guarenteed to be compatible with the device that you want to use.
        # ideally, the target_device should be provided
        return get_default_qat_qconfig(backend)
    elif symmetric_power2_quant or target_device.lower() in ('TDA4VM', 'J7ES', 'J721E', 'AM68PA'):
        '''
        This configuration will work on all our devices (but may be slightly lower accuracy compared to other options). 
        We use this if target_device is None or if target_device is explicitly specified as TDA4VM
        '''
        return get_per_tensor_symmetric_power2_qat_qconfig(backend=backend, histogram_observer=histogram_observer)
    elif per_channel_weight_quant:
        '''
        per_channel_affine is supported in devices that have MMAv2 (TDA4AL/TDA4VL/J7AEP/AM68A, TDA4VH/J7AHP/AM69A, AM62A)
        But it comes with some performance cost (lower FPS). So using it only if it is explicitly requested.
        '''
        return get_per_channel_affine_qat_qconfig(backend=backend, histogram_observer=histogram_observer)
    else:
        '''
        For devices that have MMAv2 (TDA4AL/TDA4VL/J7AEP/AM68A, TDA4VH/J7AHP/AM69A, AM62A), 
        it is possible to use per tensor affine quantization without performance cost.
        '''
        return get_per_tensor_affine_qat_qconfig(backend=backend, histogram_observer=histogram_observer)
