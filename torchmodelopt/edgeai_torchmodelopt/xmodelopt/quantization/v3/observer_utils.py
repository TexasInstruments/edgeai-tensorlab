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

import functools
import math
import random
import warnings
import torch
import torch
import torch.ao.quantization

from .... import xnn


####################################################################
def _ceil2_tensor(x):
    if x.data.abs().sum() != 0:
        x2 = xnn.layers.functional.ceil2_func(torch.abs(x))
        y = torch.sign(x) * x2
    else:
        y = x
    #
    return y


def ceil2_tensor(x):
    return xnn.layers.functional.propagate_quant_ste(x, _ceil2_tensor(x))


def ceil2_num(x):
    if x != 0:
        sign = (x>=0)*2 - 1
        x2 = math.pow(2,math.ceil(math.log2(abs(x))))
        y = sign * x2
        return y
    else:
        return x


def _adjust_qparams_power2_scale(min_val, max_val, quant_min, quant_max, scale, zero_point, eps):
    r"""Calculates the quantization parameters."""
    # make scale a power of 2 value
    scale = _ceil2_tensor(scale)
    scale = torch.max(scale, eps)
    if len(torch.unique(zero_point))>1 or torch.unique(zero_point) not in (0,127):
        # adjust the zero_point based on new scale
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
    return scale, zero_point


def _correct_min_max(min_val: torch.Tensor, max_val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    range_valid = True
    if min_val.numel() == 0 or max_val.numel() == 0:
        warnings.warn(f"must run observer before calling calculate_qparams. min_val={min_val} max_val={max_val}")
        range_valid = False
    elif torch.any(torch.isinf(min_val)) or torch.any(torch.isinf(max_val)):
        warnings.warn(f"must run observer before calling calculate_qparams. min_val={min_val} max_val={max_val}")
        range_valid = False
    elif torch.any(torch.isnan(min_val)) or torch.any(torch.isnan(max_val)):
        warnings.warn(f"invalid range: min_val={min_val} max_val={max_val}")
        range_valid = False
    elif torch.any((min_val == max_val) & (min_val == 0.0)):
        range_valid = False
    elif torch.any(min_val >= max_val):
        min_val = -torch.abs(min_val) 
        max_val = torch.abs(max_val)
    #
    return min_val, max_val, range_valid
    

def _check_min_max_valid(min_val: torch.Tensor, max_val: torch.Tensor) -> bool:
    return True
    

####################################################################
RANGE_SHRINK_PERCENTILE_DEFAULT = 0.01
RANGE_SHRINK_PERCENTILE_AGGRESSIVE = 0.1


class AdaptiveRangeShrinkObserverTypes:
    HISTOGRAM_GLOBAL = 'histogram_global'
    HISTOGRAM_RUNNINGAVG = 'histogram_runningavg'
    THREE_SIGMA_RUNNINGAVG = 'threesigma_runningavg'
    FOUR_SIGMA_RUNNINGAVG = 'foursigma_runningavg'
    DEFAULT = True # same as 'percentile'


class RangeShrinkPercentileValues:
    AGGRESSIVE = 0.1
    DEFAULT = 0.01


class AdaptiveRangeShrinkObserver(torch.ao.quantization.HistogramObserver):
    def __init__(self, *args,  factory_kwargs=None, power2_scale=False, range_max=None, fixed_range=False, 
                 range_shrink=True, dtype=torch.float32, **kwargs):
        super().__init__(*args, factory_kwargs=factory_kwargs, dtype=torch.int32, bins=1024, **kwargs)
        self.range_shrink = RangeShrinkPercentileValues.DEFAULT if isinstance(range_shrink, (bool,)) else range_shrink
        self.dtype = dtype
        self.num_batches_tracked = 0
        self.upsample_rate = 8
        self.averaging_constant = 0.01
        self.power2_scale = power2_scale
        self.range_max = range_max
        self.fixed_range = fixed_range
        self.freeze_observer = False

        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.register_buffer("min_val_hist", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val_hist", torch.tensor(float("-inf"), **factory_kwargs))

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            #
        #
    
    def forward(self, x_orig):
        if self.freeze_observer:
            return x_orig
        #
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        if torch.is_floating_point(x):
            if isinstance(self.range_shrink, (float, int)):
                # super().forward uses self.min_val and self.max_val internally
                # so copy our values into that.
                min_val = self.min_val.clone()
                max_val = self.max_val.clone()
                self.min_val.copy_(self.min_val_hist)   
                self.max_val.copy_(self.max_val_hist)
                x_o = super().forward(x)
                self.min_val_hist.copy_(self.min_val)
                self.max_val_hist.copy_(self.max_val)
                self.min_val.copy_(min_val)
                self.max_val.copy_(max_val)

                self.histogram_global_range()
            elif self.range_shrink == AdaptiveRangeShrinkObserverTypes.HISTOGRAM_RUNNINGAVG:
                x_o = super().forward(x)
                self.histogram_runningavg_range(x)
            elif self.range_shrink == AdaptiveRangeShrinkObserverTypes.FOUR_SIGMA_RUNNINGAVG:
                self.sigma_range(x, sigma_factor=4.0)
            elif self.range_shrink == AdaptiveRangeShrinkObserverTypes.THREE_SIGMA_RUNNINGAVG:
                self.sigma_range(x, sigma_factor=3.0)
            else:
                x_o = super().forward(x)
            #
        else:
            x_o = super().forward(x)
        #
        self.num_batches_tracked += 1
        return x_orig

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        # weights qparams are always symmetric and this is ensured inside the super class, no need to handle it here.
        min_val, max_val, range_valid = self._correct_min_max(min_val, max_val)
        if range_valid:
            scale, zero_point = super()._calculate_qparams(min_val, max_val)
        else:
            scale = torch.tensor(1.0, device=min_val.device)
            zero_point = torch.tensor(0, device=min_val.device, dtype=torch.int64)
        #
        if self.power2_scale:
            scale, zero_point = _adjust_qparams_power2_scale(
                min_val, max_val, self.quant_min, self.quant_max, scale, zero_point, self.eps)
        return scale, zero_point
    
    def _non_linear_param_search(self) -> tuple[torch.Tensor, torch.Tensor]:
        # called in calculate_qparams() of super class
        return self.min_val, self.max_val

    def histogram_global_range(self):
        assert self.histogram.size()[0] == self.bins, "bins mismatch"
        min_val, max_val = xnn.utils.range_from_histogram(self.histogram, self.min_val_hist, self.max_val_hist, 
                                                          bins=self.bins, range_shrink_percentile=self.range_shrink)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return self.min_val, self.max_val

    def sigma_range(self, x_orig, sigma_factor=3.0):
        min_val, max_val = xnn.utils.sigma_range(x_orig, dim=None, sigma_factor=sigma_factor)
        if torch.isnan(min_val) or torch.isnan(max_val):
            return torch.min(x_orig), torch.max(x_orig)
        self._update_runningavg_range(min_val, max_val)
        return self.min_val, self.max_val
    
    def histogram_runningavg_range(self, x_orig):
        # min_val, max_val = xnn.utils.quantile_range(x_orig, range_shrink_percentile=self.range_shrink_percentile)
        min_val, max_val = xnn.utils.histogram_range(x_orig, range_shrink_percentile=self.range_shrink_percentile)
        if torch.isnan(min_val) or torch.isnan(max_val):
            return torch.min(x_orig), torch.max(x_orig)
        self._update_runningavg_range(min_val, max_val)
        return self.min_val, self.max_val
    
    def _update_runningavg_range(self, min_val, max_val):
        if torch.isinf(self.min_val) or torch.isinf(self.max_val):
            self.min_val.copy_(min_val)
            self.max_val.copy_(max_val)
            return
        # Update the running averages
        min_val = min_val + self.averaging_constant * (min_val - self.min_val)
        max_val = max_val + self.averaging_constant * (max_val - self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
