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
        return xnn.utils.extrema_fast(x_orig, range_shrink_percentile=self.range_shrink_percentile)


class RangeShrinkHistogramObserverBase(MovingAverageRangeShrinkHistogramObserverBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, moving_average=False, **kwargs)

