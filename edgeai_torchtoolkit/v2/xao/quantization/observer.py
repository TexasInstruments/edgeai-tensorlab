
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
    def __init__(self, *args, quant_min=None, quant_max=None, dtype=None, qscheme=None, **kwargs):
        quant_min = quant_min or -128
        quant_max = quant_max or +127
        dtype = dtype or torch.qint8
        qscheme = qscheme or torch.per_tensor_symmetric
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, dtype=dtype, qscheme=qscheme, **kwargs)


class AdaptivePerChannelWeightObserver(PerChannelMinMaxObserver):
    def __init__(self, *args, quant_min=None, quant_max=None, dtype=None, qscheme=None, **kwargs):
        quant_min = quant_min or -128
        quant_max = quant_max or +127
        dtype = dtype or torch.qint8
        qscheme = qscheme or torch.per_channel_symmetric
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, dtype=dtype, qscheme=qscheme, **kwargs)


class AdaptiveActivationObserver(MovingAverageFastHistogramObserver):
    def __init__(self, *args, quant_min=None, quant_max=None, dtype=None, qscheme=None, **kwargs):
        quant_min = quant_min or 0
        quant_max = quant_max or 255
        dtype = dtype or torch.quint8
        qscheme = qscheme or torch.per_tensor_affine
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, dtype=dtype, qscheme=qscheme, **kwargs)


####################################################################
class AdaptivePower2WeightObserver(AdaptiveWeightObserver):
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


class AdaptivePower2PerChannelWeightObserver(AdaptivePerChannelWeightObserver):
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


class AdaptivePower2ActivationObserver(AdaptiveActivationObserver):
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
class AdaptiveLowBITPerChannelWeightObserver(AdaptivePerChannelWeightObserver):
    def __init__(self, *args, quant_min=None, quant_max=None, **kwargs):
        quant_min = quant_min or -7
        quant_max = quant_max or +8
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, **kwargs)


class AdaptiveLowBITActivationObserver(AdaptiveActivationObserver):
    def __init__(self, *args, quant_min=None, quant_max=None, range_shrink_percentile=observer_utils.RANGE_SHRINK_PERCENTILE_LOWBIT, **kwargs):
        quant_min = quant_min or 0
        quant_max = quant_max or 15
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, range_shrink_percentile=range_shrink_percentile, **kwargs)


####################################################################
RANGE_FIXED_VALUE = 2.0


class AdaptiveFixedRangeLowBITPerChannelWeightObserver(AdaptiveLowBITPerChannelWeightObserver):
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


class AdaptiveFixedRangeLowBITActivationObserver(AdaptiveLowBITActivationObserver):
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
                                  AdaptiveFixedRangeLowBITPerChannelWeightObserver)

ADAPTIVE_ACTIVATION_OBSERVER_TYPES = (AdaptiveActivationObserver,
                                      AdaptivePower2ActivationObserver,
                                      AdaptiveLowBITActivationObserver,
                                      AdaptiveFixedRangeLowBITActivationObserver)

ADAPTIVE_OBSERVER_TYPES = tuple(list(ADAPTIVE_WEIGHT_OBSERVER_TYPES) + list(ADAPTIVE_ACTIVATION_OBSERVER_TYPES))
