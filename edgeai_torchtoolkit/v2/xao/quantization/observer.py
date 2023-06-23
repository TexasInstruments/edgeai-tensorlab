
import torch
from torch.ao.quantization import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver


class AggressiveRangeMovingAverageMinMaxObserver(MovingAverageMinMaxObserver):
    '''
    using aggressive range may be beneficial for some cases - for example 4bit
    '''
    def __init__(self, *args, range_adjust_factor=0.75, bitwidth_adjust_factor=1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_min_orig = self.quant_min
        self.quant_max_orig = self.quant_max
        self.range_adjust_factor = range_adjust_factor
        self.bitwidth_adjust_factor = bitwidth_adjust_factor

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


class AggressiveRangeMovingAveragePerChannelMinMaxObserver(MovingAveragePerChannelMinMaxObserver):
    '''
    using aggressive range may be beneficial for some cases - for example 4bit
    '''
    def __init__(self, *args, range_adjust_factor=0.75, bitwidth_adjust_factor=1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_min_orig = self.quant_min
        self.quant_max_orig = self.quant_max
        self.range_adjust_factor = range_adjust_factor
        self.bitwidth_adjust_factor = bitwidth_adjust_factor

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


aggressive_range_observers_types = (AggressiveRangeMovingAverageMinMaxObserver,
                                    AggressiveRangeMovingAveragePerChannelMinMaxObserver)
