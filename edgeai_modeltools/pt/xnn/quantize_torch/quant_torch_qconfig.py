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

import warnings
import random
import torch

######################################################################
def get_custom_qconfig_observers(backend='fbgemm', bitwidth=8, histogram=True, symmetric=False,  per_channel=False,
                       power2_weight_range=False, power2_activation_range=False, reduce_range=False):
    assert bitwidth==8, 'Only 8-bit quantization is supported with PyTrch native quantization'
    assert backend == 'fbgemm' or backend == 'qnnpack', f'invalid quantization backend {backend}'
    power2_scale = (power2_weight_range or power2_activation_range)
    if backend == 'qnnpack':
        assert per_channel==False, 'per_channel quantization is not supported for qnnpack backend'
        assert reduce_range==False, 'reduce_range is not required for qnnpack backend'
    else:
        # fbgemm provides fast matrix multiplications: https://engineering.fb.com/ml-applications/fbgemm/
        # can use int16 accumulation for speedup followed by a sparse matrix multiplication and accumulation to handle outliers
        # avoiding saturation can help to speedup (as the second sparse operation will be reduced)
        # but it seems the sparse multiply and acc is not happening since we see an accuracy degradation with tight ranges.
        assert power2_scale or reduce_range, 'reduce_range is required for fbgemm backend. ' \
                'but if using power2_scale, it reduces range already - then reduce_range may not be required'
    #
    if symmetric or power2_scale:
        # weight range is always symmetric, but can be per_channel or per_tensor
        WeightObserver = PerChannelMinMaxObserverCustom if per_channel else MinMaxObserverCustom
        weight_observer = WeightObserver.with_args(symmetric=True, per_channel=per_channel,\
                                power2_scale=power2_weight_range, dtype=torch.qint8, reduce_range=reduce_range)
        # activation scale is always per tensor, but can be affine or symmetric
        ActivationObserver = HistogramObserverCustom if histogram else MovingAverageMinMaxObserverCustom
        activation_observer = ActivationObserver.with_args(symmetric=symmetric, per_channel=False,\
                                power2_scale=power2_activation_range, dtype=torch.quint8, reduce_range=reduce_range)
    else:
        # reduce range is done differently from default torch way - we give preference to activation here.
        WeightObserver = torch.quantization.PerChannelMinMaxObserver if per_channel else torch.quantization.MinMaxObserver
        weight_observer = WeightObserver.with_args(reduce_range=reduce_range)
        ActivationObserver = torch.quantization.HistogramObserver if histogram else torch.quantization.MovingAverageMinMaxObserver
        activation_observer = ActivationObserver.with_args(reduce_range=False)
    #
    return weight_observer, activation_observer


######################################################################
def get_custom_qconfig(backend='fbgemm', bitwidth=8, symmetric=False,  per_channel=False,
                       power2_weight_range=False, power2_activation_range=False,
                       reduce_range=False, histogram=True):
    weight_observer, activation_observer = get_custom_qconfig_observers(backend=backend, bitwidth=bitwidth, symmetric=symmetric,
                        per_channel=per_channel, power2_weight_range=power2_weight_range, power2_activation_range=power2_activation_range,
                        reduce_range=reduce_range, histogram=histogram)
    qconfig = torch.quantization.QConfig(activation=activation_observer, weight=weight_observer)
    return qconfig

######################################################################
def get_custom_qconfig_with_fakequantize(backend='fbgemm', bitwidth=8, symmetric=False,  per_channel=False,
                       power2_weight_range=False, power2_activation_range=False,
                       reduce_range=False, quantize_weights=True, quantize_activation=True, histogram=True):
    weight_observer, activation_observer = get_custom_qconfig_observers(backend=backend, bitwidth=bitwidth, symmetric=symmetric,
                        per_channel=per_channel, power2_weight_range=power2_weight_range,
                        power2_activation_range=power2_activation_range,
                        reduce_range=reduce_range, histogram=histogram)
    # reduce range is done differently from default torch way - we give preference to activation here.
    if quantize_weights:
        weight_observer = torch.quantization.FakeQuantize.with_args(observer=weight_observer,
                                        quant_min=-128, quant_max=127, reduce_range=False)
    #
    if quantize_activation:
        activation_observer = torch.quantization.FakeQuantize.with_args(observer=activation_observer,
                                        quant_min=0, quant_max=255, reduce_range=reduce_range)
    #
    qconfig = torch.quantization.QConfig(activation=activation_observer, weight=weight_observer)
    return qconfig


######################################################################
def calculate_qparams_adaptive(self, min_val, max_val):
    # to handle the per_channel_case
    min_val2 = torch.min(min_val)
    # make qscheme depend on min_val - this is because we want to use quint8 for relu output
    if self.symmetric and (min_val2 < 0):
        self.qscheme = torch.per_tensor_symmetric
    else:
        self.qscheme = torch.per_tensor_affine
    #
    scale, zero_point = self._calculate_qparams(min_val, max_val)
    scale = ceil2(scale) if self.power2_scale else scale
    return scale, zero_point


######################################################################
class HistogramObserverCustom(torch.quantization.HistogramObserver):
    def __init__(self, symmetric, per_channel, power2_scale, dtype=torch.quint8, reduce_range=False, fast_mode=False):
        if fast_mode:
            super().__init__(dtype=dtype, reduce_range=reduce_range, bins=512, upsample_rate=32)
        else:
            super().__init__(dtype=dtype, reduce_range=reduce_range)
        #
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.power2_scale = power2_scale
        self.fast_mode = fast_mode
        assert not per_channel, 'per_channel is not supported for this observer right now. use minmax observer for that.'

    @torch.jit.export
    def calculate_qparams(self):
        is_uninitialized = (self.min_val == float('inf') and self.max_val == float('-inf'))
        if is_uninitialized:
            warnings.warn("must run observer before calling calculate_qparams. Returning default scale and zero point ")
            return torch.tensor([1.0]), torch.tensor([0])
        assert self.bins == len(self.histogram), \
            ("The number of bins in histogram should be equal to the number of bins supplied while making this observer")
        # max must be greater than min - ensure this
        same_values = False
        if self.min_val.numel() > 0 and self.max_val.numel() > 0:
            same_values = (self.min_val.item() == self.max_val.item())
        #
        if same_values:
            self.qscheme = torch.per_tensor_affine
            return torch.tensor([1.0]), torch.tensor([0])
        #
        new_min, new_max = self._non_linear_param_search()
        scale, zero_point = calculate_qparams_adaptive(self, new_min, new_max)
        return scale, zero_point

    def forward(self, x_orig):
        # type: (Tensor) -> Tensor
        # Note: enable this code to do range calibration only in training mode
        # if not self.training:
        #     return x_orig
        # #
        try:
            x_subsampled = x_orig
            if self.fast_mode:
                fast_stride = 2
                fast_stride2 = fast_stride * 2
                if len(x_orig.size()) == 4 and (x_orig.size(2) > fast_stride2) and (x_orig.size(3) > fast_stride2):
                    r_start = random.randint(0, fast_stride - 1)
                    c_start = random.randint(0, fast_stride - 1)
                    x_subsampled = x_orig[..., r_start::fast_stride, c_start::fast_stride]
                #
            #
            super().forward(x_subsampled)
            return x_orig
        except:
            return x_orig
        #


class MovingAverageMinMaxObserverCustom(torch.quantization.MovingAverageMinMaxObserver):
    def __init__(self, symmetric, per_channel, power2_scale, dtype=torch.quint8, reduce_range=False):
        super().__init__(dtype=dtype, reduce_range=reduce_range)
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.power2_scale = power2_scale
        # Note: enable this code to do range calibration only in training mode and with a diminishing factor
        # self.counter = 0.0
        # self.averaging_constant = 1.0
        assert not per_channel, 'per_channel is not supported for this observer right now. use minmax observer for that.'

    @torch.jit.export
    def calculate_qparams(self):
        return calculate_qparams_adaptive(self, self.min_val, self.max_val)

    def forward(self, x_orig):
        # Note: enable this code to do range calibration only in training mode and with a diminishing factor
        # if self.training:
        #     self.averaging_constant = 1.0 / (self.counter + 1.0)
        #     self.counter += 1.0
        # #
        return super().forward(x_orig)


######################################################################
class MinMaxObserverCustom(torch.quantization.MinMaxObserver):
    def __init__(self, symmetric, per_channel, power2_scale, dtype=torch.quint8, reduce_range=False):
        super().__init__(dtype=dtype, reduce_range=reduce_range)
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.power2_scale = power2_scale
        assert symmetric, 'this observer is only recommended for weights - which needs symmetric'
        assert not per_channel, 'per_channel is not supported for this observer'

    @torch.jit.export
    def calculate_qparams(self):
        return calculate_qparams_adaptive(self, self.min_val, self.max_val)


class PerChannelMinMaxObserverCustom(torch.quantization.PerChannelMinMaxObserver):
    def __init__(self, symmetric, per_channel, power2_scale, dtype=torch.quint8, reduce_range=False):
        super().__init__(dtype=dtype, reduce_range=reduce_range)
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.power2_scale = power2_scale
        assert symmetric, 'this observer is only recommended for weights - which needs symmetric'
        assert per_channel, 'per_channel must be set for this observer'

    @torch.jit.export
    def calculate_qparams(self):
        return calculate_qparams_adaptive(self, self.min_vals, self.max_vals)


######################################################################
def ceil2(x):
    eps = 126.9 / 128.0
    y = torch.pow(2, torch.ceil(torch.log2(x*eps)))
    return y


def floor2(x):
    eps = 128.0 / 126.9
    y = torch.pow(2, torch.floor(torch.log2(x*eps)))
    return y
