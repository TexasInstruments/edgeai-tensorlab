
import functools
import math
import random
import torch
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, HistogramObserver, \
    MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from ....v1 import xnn


####################################################################
def ceil2_tensor(x):
    with torch.no_grad():
        if x.data.abs().sum() != 0:
            x2 = xnn.layers.functional.ceil2_func(torch.abs(x))
            y = torch.sign(x) * x2
        else:
            y = x
        #
    #
    return y


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
# not as good as our version down below
class FastMSEHistogramObserver(HistogramObserver):
    def __init__(self, *args, range_shrink_percentile=None, **kwargs):
        super().__init__(*args, bins=256, upsample_rate=16, **kwargs)


####################################################################
RANGE_SHRINK_PERCENTILE_DEFAULT = 0.01
RANGE_SHRINK_PERCENTILE_LOWBIT = 1.0 #0.1


####################################################################
class MovingAverageFastHistogramObserver(MinMaxObserver):
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
        super(MovingAverageFastHistogramObserver, self).__init__(
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


class FastHistogramObserver(MovingAverageFastHistogramObserver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, moving_average=False, **kwargs)


####################################################################
class AdaptiveWeightObserver(FastHistogramObserver):
    pass


class AdaptivePerChannelWeightObserver(PerChannelMinMaxObserver):
    pass


class AdaptiveActivationObserver(MovingAverageFastHistogramObserver):
    pass


####################################################################
class AdaptiveLowBITPerChannelWeightObserver(PerChannelMinMaxObserver):
    pass


class AdaptiveLowBITActivationObserver(MovingAverageFastHistogramObserver):
    pass


####################################################################
class AdaptivePower2WeightObserver(FastHistogramObserver):
    '''
    Create a subclass, just to distinguish between the ones used for activation and weight
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_min_orig = self.quant_min
        self.quant_max_orig = self.quant_max

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        self.quant_min = self.ceil2_num(self.quant_min_orig)
        self.quant_max = self.ceil2_num(self.quant_max_orig)
        qparams = super()._calculate_qparams(ceil2_tensor(min_val), ceil2_tensor(max_val))
        self.quant_min, self.quant_max = self.quant_min_orig, self.quant_max_orig
        return qparams


class AdaptivePower2PerChannelWeightObserver(PerChannelMinMaxObserver):
    '''
    Create a subclass, just to distinguish between the ones used for activation and weight
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_min_orig = self.quant_min
        self.quant_max_orig = self.quant_max

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        self.quant_min = self.ceil2_num(self.quant_min_orig)
        self.quant_max = self.ceil2_num(self.quant_max_orig)
        qparams = super()._calculate_qparams(ceil2_tensor(min_val), ceil2_tensor(max_val))
        self.quant_min, self.quant_max = self.quant_min_orig, self.quant_max_orig
        return qparams


class AdaptivePower2ActivationObserver(MovingAverageFastHistogramObserver):
    '''
    Create a subclass, just to distinguish between the ones used for activation and weight
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, range_shrink_percentile=RANGE_SHRINK_PERCENTILE_DEFAULT, **kwargs)
        self.quant_min_orig = self.quant_min
        self.quant_max_orig = self.quant_max

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        self.quant_min = self.ceil2_num(self.quant_min_orig)
        self.quant_max = self.ceil2_num(self.quant_max_orig)
        qparams = super()._calculate_qparams(ceil2_tensor(min_val), ceil2_tensor(max_val))
        self.quant_min, self.quant_max = self.quant_min_orig, self.quant_max_orig
        return qparams


####################################################################
ADAPTIVE_WEIGHT_OBSERVER_TYPES = (AdaptiveWeightObserver,
                                  AdaptivePerChannelWeightObserver,
                                  AdaptivePower2WeightObserver,
                                  AdaptivePower2PerChannelWeightObserver,
                                  AdaptiveLowBITPerChannelWeightObserver)

ADAPTIVE_ACTIVATION_OBSERVER_TYPES = (AdaptiveActivationObserver,
                                      AdaptivePower2ActivationObserver,
                                      AdaptiveLowBITActivationObserver)

ADAPTIVE_OBSERVER_TYPES = tuple(list(ADAPTIVE_WEIGHT_OBSERVER_TYPES) + list(ADAPTIVE_ACTIVATION_OBSERVER_TYPES))
