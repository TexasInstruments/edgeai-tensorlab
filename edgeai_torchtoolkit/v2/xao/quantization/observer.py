
import torch
from torch.ao.quantization import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver, HistogramObserver


class AggressiveRangeHistogramObserver(HistogramObserver):
    '''
    using aggressive range may be beneficial for some cases - for example 4bit
    '''
    def __init__(self, *args, range_shrink_factor=1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_min_orig = self.quant_min
        self.quant_max_orig = self.quant_max
        self.range_shrink_factor = range_shrink_factor

    @classmethod
    def set_aggressive_range(cls, value=True):
        cls.aggressive_range = value

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        if self.aggressive_range:
            self.quant_min = self.quant_min_orig
            self.quant_max = self.quant_max_orig
        else:
            self.quant_min = self.quant_min_orig*16
            self.quant_max = self.quant_max_orig*16
        #
        return super()._calculate_qparams(min_val*self.range_shrink_factor, max_val*self.range_shrink_factor)


class AggressiveRangeMovingAverageMinMaxObserver(MovingAverageMinMaxObserver):
    '''
    using aggressive range may be beneficial for some cases - for example 4bit
    '''
    def __init__(self, *args, range_shrink_factor=0.75, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_min_orig = self.quant_min
        self.quant_max_orig = self.quant_max
        self.range_shrink_factor = range_shrink_factor

    @classmethod
    def set_aggressive_range(cls, value=True):
        cls.aggressive_range = value

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        if self.aggressive_range:
            self.quant_min = self.quant_min_orig
            self.quant_max = self.quant_max_orig
        else:
            self.quant_min = self.quant_min_orig*16
            self.quant_max = self.quant_max_orig*16
        #
        return super()._calculate_qparams(min_val*self.range_shrink_factor, max_val*self.range_shrink_factor)


class AggressiveRangeMovingAveragePerChannelMinMaxObserver(MovingAveragePerChannelMinMaxObserver):
    '''
    using aggressive range may be beneficial for some cases - for example 4bit
    '''
    def __init__(self, *args, range_shrink_factor=0.75, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_min_orig = self.quant_min
        self.quant_max_orig = self.quant_max
        self.range_shrink_factor = range_shrink_factor

    @classmethod
    def set_aggressive_range(cls, value=True):
        cls.aggressive_range = value

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        if self.aggressive_range:
            self.quant_min = self.quant_min_orig
            self.quant_max = self.quant_max_orig
        else:
            self.quant_min = self.quant_min_orig*16
            self.quant_max = self.quant_max_orig*16
        #
        return super()._calculate_qparams(min_val*self.range_shrink_factor, max_val*self.range_shrink_factor)


aggressive_range_observers_types = (AggressiveRangeHistogramObserver,
                                    AggressiveRangeMovingAverageMinMaxObserver,
                                    AggressiveRangeMovingAveragePerChannelMinMaxObserver)
