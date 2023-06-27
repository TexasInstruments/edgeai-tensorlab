
import functools
import math
import random
import torch
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, HistogramObserver, \
    MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from ....v1 import xnn
from . import observer_utils


####################################################################
# select which histogram observer to use in this file as the base

# FastHistogramObserver = observer_utils.MSEHistogramObserverBase
# MovingAverageFastHistogramObserver = observer_utils.MovingAverageMSEHistogramObserverBase

FastHistogramObserver = observer_utils.RangeShrinkHistogramObserverBase
MovingAverageFastHistogramObserver = observer_utils.MovingAverageRangeShrinkHistogramObserverBase


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, range_shrink_percentile=observer_utils.RANGE_SHRINK_PERCENTILE_LOWBIT, **kwargs)


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
        self.quant_min = observer_utils.ceil2_num(self.quant_min_orig)
        self.quant_max = observer_utils.ceil2_num(self.quant_max_orig)
        qparams = super()._calculate_qparams(observer_utils.ceil2_tensor(min_val), observer_utils.ceil2_tensor(max_val))
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
        self.quant_min = observer_utils.ceil2_num(self.quant_min_orig)
        self.quant_max = observer_utils.ceil2_num(self.quant_max_orig)
        qparams = super()._calculate_qparams(observer_utils.ceil2_tensor(min_val), observer_utils.ceil2_tensor(max_val))
        self.quant_min, self.quant_max = self.quant_min_orig, self.quant_max_orig
        return qparams


class AdaptivePower2ActivationObserver(MovingAverageFastHistogramObserver):
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
        self.quant_min = observer_utils.ceil2_num(self.quant_min_orig)
        self.quant_max = observer_utils.ceil2_num(self.quant_max_orig)
        qparams = super()._calculate_qparams(observer_utils.ceil2_tensor(min_val), observer_utils.ceil2_tensor(max_val))
        self.quant_min, self.quant_max = self.quant_min_orig, self.quant_max_orig
        return qparams


####################################################################
RANGE_FIXED_VALUE = 2.0


class AdaptiveFixedRangePerChannelWeightObserver(AdaptivePower2PerChannelWeightObserver):
    def __init__(self, *args, range_val=RANGE_FIXED_VALUE, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.range_val = range_val

    def forward(self, x_orig):
        x_orig = super().forward(x_orig)
        signed_range = torch.min(self.min_val.detach()).item() < 0.0
        min_val = (-self.range_val) if signed_range else 0.0
        max_val = (+self.range_val) if signed_range else (+self.range_val)
        self.min_val.fill_(min_val)
        self.max_val.fill_(max_val)
        return x_orig


class AdaptiveFixedRangeActivationObserver(AdaptivePower2ActivationObserver):
    def __init__(self, *args, range_val=RANGE_FIXED_VALUE, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.range_val = range_val

    def forward(self, x_orig):
        x_orig = super().forward(x_orig)
        signed_range = torch.min(self.min_val.detach()).item() < 0.0
        min_val = (-self.range_val) if signed_range else 0.0
        max_val = (+self.range_val) if signed_range else (+self.range_val)
        self.min_val.fill_(min_val)
        self.max_val.fill_(max_val)
        return x_orig


####################################################################
ADAPTIVE_WEIGHT_OBSERVER_TYPES = (AdaptiveWeightObserver,
                                  AdaptivePerChannelWeightObserver,
                                  AdaptivePower2WeightObserver,
                                  AdaptivePower2PerChannelWeightObserver,
                                  AdaptiveLowBITPerChannelWeightObserver,
                                  AdaptiveFixedRangePerChannelWeightObserver)

ADAPTIVE_ACTIVATION_OBSERVER_TYPES = (AdaptiveActivationObserver,
                                      AdaptivePower2ActivationObserver,
                                      AdaptiveLowBITActivationObserver,
                                      AdaptiveFixedRangeActivationObserver)

ADAPTIVE_OBSERVER_TYPES = tuple(list(ADAPTIVE_WEIGHT_OBSERVER_TYPES) + list(ADAPTIVE_ACTIVATION_OBSERVER_TYPES))
