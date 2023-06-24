
import math
import torch
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, HistogramObserver, \
    MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from ....v1 import xnn


class FastHistogramObserver(MinMaxObserver):
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
        moving_average=True,
        **kwargs
    ) -> None:
        self.averaging_constant = averaging_constant
        super(FastHistogramObserver, self).__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            **kwargs
        )
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
        return xnn.utils.extrema_fast(x_orig, range_shrink_percentile=0.01)


####################################################################
class AdaptiveWeightObserver(MinMaxObserver):
    # not using MovingAverageMinMaxObserver - weight may not need it
    '''
    Create a subclass, just to distinguish between the ones used for activation and weight
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_min_orig = self.quant_min
        self.quant_max_orig = self.quant_max
        self.range_adjust_factor = 1.0
        self.bitwidth_adjust_factor = 1.0

    def set_range_adjust_factor(self, value=1.0):
        self.range_adjust_factor = value

    def set_bitwidth_adjust_factor(self, value=1.0):
        self.bitwidth_adjust_factor = value

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        self.quant_min = int(round(self.quant_min_orig*self.bitwidth_adjust_factor))
        self.quant_max = int(round(self.quant_max_orig*self.bitwidth_adjust_factor))
        return super()._calculate_qparams(min_val*self.range_adjust_factor, max_val*self.range_adjust_factor)


class AdaptiveActivationObserver(FastHistogramObserver):
    '''
    using aggressive range may be beneficial for some cases - for example 4bit
    this observer is not specific to 4bit - but the name is just indicative this it is very aggressive
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_min_orig = self.quant_min
        self.quant_max_orig = self.quant_max
        self.range_adjust_factor = 1.0
        self.bitwidth_adjust_factor = 1.0

    def set_range_adjust_factor(self, value=1.0):
        self.range_adjust_factor = value

    def set_bitwidth_adjust_factor(self, value=1.0):
        self.bitwidth_adjust_factor = value

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        self.quant_min = int(round(self.quant_min_orig*self.bitwidth_adjust_factor))
        self.quant_max = int(round(self.quant_max_orig*self.bitwidth_adjust_factor))
        return super()._calculate_qparams(min_val*self.range_adjust_factor, max_val*self.range_adjust_factor)


####################################################################
class AdaptivePerChannelWeightObserver(PerChannelMinMaxObserver):
    # not using MovingAveragePerChannelMinMaxObserver - weight may not need it
    '''
    using aggressive range may be beneficial for some cases - for example 4bit
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_min_orig = self.quant_min
        self.quant_max_orig = self.quant_max
        self.range_adjust_factor = 1.0
        self.bitwidth_adjust_factor = 1.0

    def set_range_adjust_factor(self, value=1.0):
        self.range_adjust_factor = value

    def set_bitwidth_adjust_factor(self, value=1.0):
        self.bitwidth_adjust_factor = value

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        self.quant_min = int(round(self.quant_min_orig*self.bitwidth_adjust_factor))
        self.quant_max = int(round(self.quant_max_orig*self.bitwidth_adjust_factor))
        return super()._calculate_qparams(min_val*self.range_adjust_factor, max_val*self.range_adjust_factor)


####################################################################
class AdaptivePower2WeightObserver(MinMaxObserver):
    '''
    Create a subclass, just to distinguish between the ones used for activation and weight
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_min_orig = self.quant_min
        self.quant_max_orig = self.quant_max
        self.range_adjust_factor = 1.0
        self.bitwidth_adjust_factor = 1.0

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        self.quant_min = self.ceil2_num(self.quant_min_orig*self.bitwidth_adjust_factor)
        self.quant_max = self.ceil2_num(self.quant_max_orig*self.bitwidth_adjust_factor)
        qparams = super()._calculate_qparams(self.ceil2_tensor(min_val*self.range_adjust_factor),
                                             self.ceil2_tensor(max_val*self.range_adjust_factor))
        self.quant_min, self.quant_max = self.quant_min_orig, self.quant_max_orig
        return qparams

    @torch.jit.export
    def ceil2_tensor(self, x):
        with torch.no_grad():
            if x.data.abs().sum() != 0:
                x2 = xnn.layers.functional.ceil2_func(torch.abs(x))
                y = torch.sign(x) * x2
            else:
                y = x
            #
        #
        return y

    def ceil2_num(self, x):
        if x != 0:
            sign = (x>=0)*2 - 1
            x2 = math.pow(2,math.ceil(math.log2(abs(x))))
            y = sign * x2
            return y
        else:
            return x


class AdaptivePower2ActivationObserver(FastHistogramObserver):
    '''
    Create a subclass, just to distinguish between the ones used for activation and weight
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_min_orig = self.quant_min
        self.quant_max_orig = self.quant_max
        self.range_adjust_factor = 1.0
        self.bitwidth_adjust_factor = 1.0

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        self.quant_min = self.ceil2_num(self.quant_min_orig*self.bitwidth_adjust_factor)
        self.quant_max = self.ceil2_num(self.quant_max_orig*self.bitwidth_adjust_factor)
        qparams = super()._calculate_qparams(self.ceil2_tensor(min_val*self.range_adjust_factor),
                                             self.ceil2_tensor(max_val*self.range_adjust_factor))
        self.quant_min, self.quant_max = self.quant_min_orig, self.quant_max_orig
        return qparams

    @torch.jit.export
    def ceil2_tensor(self, x):
        with torch.no_grad():
            if x.data.abs().sum() != 0:
                x2 = xnn.layers.functional.ceil2_func(torch.abs(x))
                y = torch.sign(x) * x2
            else:
                y = x
            #
        #
        return y

    def ceil2_num(self, x):
        if x != 0:
            sign = (x>=0)*2 - 1
            x2 = math.pow(2,math.ceil(math.log2(abs(x))))
            y = sign * x2
            return y
        else:
            return x


####################################################################
ADAPTIVE_WEIGHT_OBSERVER_TYPES = (AdaptiveWeightObserver,
                                  AdaptivePerChannelWeightObserver,
                                  AdaptivePower2WeightObserver)

ADAPTIVE_ACTIVATION_OBSERVER_TYPES = (AdaptiveActivationObserver,
                                      AdaptivePower2ActivationObserver)
