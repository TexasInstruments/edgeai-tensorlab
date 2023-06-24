
import math
import torch
from torch.ao.quantization import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from ....v1 import xnn


class RangeAdjustMovingAverageMinMaxObserver(MovingAverageMinMaxObserver):
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
        self.quant_min = int(round(self.quant_min_orig * self.bitwidth_adjust_factor))
        self.quant_max = int(round(self.quant_max_orig * self.bitwidth_adjust_factor))
        return super()._calculate_qparams(min_val*self.range_adjust_factor, max_val*self.range_adjust_factor)


class RangeAdjustMovingAveragePerChannelMinMaxObserver(MovingAveragePerChannelMinMaxObserver):
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
        self.quant_min = int(round(self.quant_min_orig * self.bitwidth_adjust_factor))
        self.quant_max = int(round(self.quant_max_orig * self.bitwidth_adjust_factor))
        return super()._calculate_qparams(min_val*self.range_adjust_factor, max_val*self.range_adjust_factor)


class Power2MovingAverageMinMaxObserver(MovingAverageMinMaxObserver):
    '''
    using aggressive range may be beneficial for some cases - for example 4bit
    '''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_min_orig = self.quant_min
        self.quant_max_orig = self.quant_max

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        self.quant_min, self.quant_max = self.ceil2_num(self.quant_min_orig), self.ceil2_num(self.quant_max_orig)
        qparams = super()._calculate_qparams(self.ceil2_tensor(min_val), self.ceil2_tensor(max_val))
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


RANGE_ADJUST_OBSERVER_TYPES = (RangeAdjustMovingAverageMinMaxObserver,
                               RangeAdjustMovingAveragePerChannelMinMaxObserver)
