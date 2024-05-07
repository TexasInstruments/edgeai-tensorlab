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
import torch
import torch
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, HistogramObserver, \
    MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
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


####################################################################
# histogram observer from torch.ao.quantization
# (MSE based and includes merging of histograms across iterations)
class MovingAverageMSEHistogramObserverBase(HistogramObserver):
    def __init__(self, *args, range_shrink_percentile=None, **kwargs):
        super().__init__(*args, bins=256, upsample_rate=16, **kwargs)

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        fast_mode = True
        fast_stride = 2
        fast_stride2 = fast_stride * 2
        if fast_mode and len(x_orig.size()) == 4 and (x_orig.size(2) > fast_stride2) and (x_orig.size(3) > fast_stride2):
            r_start = random.randint(0, fast_stride - 1)
            c_start = random.randint(0, fast_stride - 1)
            src = x_orig[..., r_start::fast_stride, c_start::fast_stride]
        else:
            src = x_orig
        #
        super().forward(src)
        return x_orig


class MSEHistogramObserverBase(MovingAverageMSEHistogramObserverBase):
    def __init__(self, *args, range_shrink_percentile=None, **kwargs):
        super().__init__(*args, bins=256, upsample_rate=16, **kwargs)

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        self.min_val = float("inf")
        self.max_val = float("-inf")
        super().forward(x_orig)


####################################################################
RANGE_SHRINK_PERCENTILE_DEFAULT = 0.01
RANGE_SHRINK_PERCENTILE_LOWBIT = 0.1


class MovingAverageRangeShrinkHistogramObserverBase(MinMaxObserver):
    # histogram observer may improve accuracy.
    # default histogram observer in torch.ao.quantization is too slow - so using a custom one
    def __init__(
        self,
        averaging_constant=0.01,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        range_shrink_percentile=RANGE_SHRINK_PERCENTILE_DEFAULT,
        moving_average=True,
        **kwargs
    ) -> None:
        self.averaging_constant = averaging_constant
        super(MovingAverageRangeShrinkHistogramObserverBase, self).__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            **kwargs
        )
        self.range_shrink_percentile = range_shrink_percentile
        self.moving_average = moving_average

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        if (not self.moving_average) or (min_val == float("inf") and max_val == float("-inf")):
            min_val, max_val = self.histogram_range(x)
        else:
            min_val_cur, max_val_cur = self.histogram_range(x)
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    def histogram_range(self, x_orig):
        # quantile_l = self.range_shrink_percentile/100.0
        # quantile_h = 1.0 - quantile_l
        # r_min = torch.quantile(x_orig, quantile_l)
        # r_max = torch.quantile(x_orig, quantile_h)
        # r_min_max = (r_min, r_max)
        r_min_max = xnn.utils.extrema_fast(x_orig, range_shrink_percentile=self.range_shrink_percentile)
        return r_min_max


class RangeShrinkHistogramObserverBase(MovingAverageRangeShrinkHistogramObserverBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, moving_average=False, **kwargs)

