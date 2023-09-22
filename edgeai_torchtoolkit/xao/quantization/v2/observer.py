
import functools
import math
import random
import torch
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, HistogramObserver, \
    MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver
from .... import xnn
from . import observer_utils


####################################################################
# select which histogram observer to use in this file as the base

# FastHistogramObserver = observer_utils.MSEHistogramObserverBase
# MovingAverageFastHistogramObserver = observer_utils.MovingAverageMSEHistogramObserverBase

FastHistogramObserver = observer_utils.RangeShrinkHistogramObserverBase
MovingAverageFastHistogramObserver = observer_utils.MovingAverageRangeShrinkHistogramObserverBase


####################################################################
class AdaptiveWeightObserver(FastHistogramObserver):
    def __init__(self, *args, quant_min=None, quant_max=None, dtype=None, qscheme=None, power2=False, range_val=None, **kwargs):
        quant_min = quant_min or -128
        quant_max = quant_max or +127
        dtype = dtype or torch.qint8
        qscheme = qscheme or torch.per_tensor_symmetric
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, dtype=dtype, qscheme=qscheme, **kwargs)
        self.power2 = power2
        self.range_val = range_val
        self.quant_min_orig = self.quant_min
        self.quant_max_orig = self.quant_max

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        if not self.power2:
            return super()._calculate_qparams(min_val, max_val)
        else:
            self.quant_min = observer_utils.ceil2_num(self.quant_min_orig)
            self.quant_max = observer_utils.ceil2_num(self.quant_max_orig)
            qparams = super()._calculate_qparams(observer_utils.ceil2_tensor(min_val), observer_utils.ceil2_tensor(max_val))
            self.quant_min, self.quant_max = self.quant_min_orig, self.quant_max_orig
            return qparams

    def forward(self, x_orig):
        x_orig = super().forward(x_orig)
        if self.range_val is not None:
            signed_range = torch.min(self.min_val.detach()).item() < 0.0
            min_val = (-self.range_val) if signed_range else 0.0
            max_val = (+self.range_val) if signed_range else (+self.range_val)
            # self.min_val.fill_(min_val)
            # self.max_val.fill_(max_val)
            self.min_val = torch.clamp(self.min_val, min=min_val, max=0.0)
            self.max_val = torch.clamp(self.max_val, min=0.0, max=max_val)
        #
        return x_orig


class AdaptivePerChannelWeightObserver(PerChannelMinMaxObserver):
    def __init__(self, *args, quant_min=None, quant_max=None, dtype=None, qscheme=None, power2=False, range_val=None, **kwargs):
        quant_min = quant_min or -128
        quant_max = quant_max or +127
        dtype = dtype or torch.qint8
        qscheme = qscheme or torch.per_channel_symmetric
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, dtype=dtype, qscheme=qscheme, **kwargs)
        self.power2 = power2
        self.range_val = range_val
        self.quant_min_orig = self.quant_min
        self.quant_max_orig = self.quant_max

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        if not self.power2:
            return super()._calculate_qparams(min_val, max_val)
        else:
            self.quant_min = observer_utils.ceil2_num(self.quant_min_orig)
            self.quant_max = observer_utils.ceil2_num(self.quant_max_orig)
            qparams = super()._calculate_qparams(observer_utils.ceil2_tensor(min_val), observer_utils.ceil2_tensor(max_val))
            self.quant_min, self.quant_max = self.quant_min_orig, self.quant_max_orig
            return qparams

    def forward(self, x_orig):
        x_orig = super().forward(x_orig)
        if self.range_val is not None:
            signed_range = torch.min(self.min_val.detach()).item() < 0.0
            min_val = (-self.range_val) if signed_range else 0.0
            max_val = (+self.range_val) if signed_range else (+self.range_val)
            # self.min_val.fill_(min_val)
            # self.max_val.fill_(max_val)
            self.min_val = torch.clamp(self.min_val, min=min_val, max=0.0)
            self.max_val = torch.clamp(self.max_val, min=0.0, max=max_val)
        #
        return x_orig


class AdaptiveActivationObserver(MovingAverageFastHistogramObserver):
    def __init__(self, *args, quant_min=None, quant_max=None, dtype=None, qscheme=None, power2=False, range_val=None, **kwargs):
        quant_min = quant_min or 0
        quant_max = quant_max or 255
        dtype = dtype or torch.quint8
        qscheme = qscheme or torch.per_tensor_affine
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, dtype=dtype, qscheme=qscheme, **kwargs)
        self.power2 = power2
        self.range_val = range_val
        self.quant_min_orig = self.quant_min
        self.quant_max_orig = self.quant_max

    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        if not self.power2:
            return super()._calculate_qparams(min_val, max_val)
        else:
            self.quant_min = observer_utils.ceil2_num(self.quant_min_orig)
            self.quant_max = observer_utils.ceil2_num(self.quant_max_orig)
            qparams = super()._calculate_qparams(observer_utils.ceil2_tensor(min_val), observer_utils.ceil2_tensor(max_val))
            self.quant_min, self.quant_max = self.quant_min_orig, self.quant_max_orig
            return qparams

    def forward(self, x_orig):
        x_orig = super().forward(x_orig)
        if self.range_val is not None:
            signed_range = torch.min(self.min_val.detach()).item() < 0.0
            min_val = (-self.range_val) if signed_range else 0.0
            max_val = (+self.range_val) if signed_range else (+self.range_val)
            # self.min_val.fill_(min_val)
            # self.max_val.fill_(max_val)
            self.min_val = torch.clamp(self.min_val, min=min_val, max=0.0)
            self.max_val = torch.clamp(self.max_val, min=0.0, max=max_val)
        #
        return x_orig


####################################################################
class AdaptivePower2WeightObserver(AdaptiveWeightObserver):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, power2=True, **kwargs)


class AdaptivePower2PerChannelWeightObserver(AdaptivePerChannelWeightObserver):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, power2=True, **kwargs)


class AdaptivePower2ActivationObserver(AdaptiveActivationObserver):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, power2=True, **kwargs)


####################################################################
class AdaptiveLowBITPerChannelWeightObserver(AdaptivePerChannelWeightObserver):
    def __init__(self, *args, quant_min=None, quant_max=None, **kwargs):
        quant_min = quant_min or -8
        quant_max = quant_max or +7
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, **kwargs)


class AdaptiveLowBITActivationObserver(AdaptiveActivationObserver):
    def __init__(self, *args, quant_min=None, quant_max=None, range_shrink_percentile=observer_utils.RANGE_SHRINK_PERCENTILE_LOWBIT, **kwargs):
        quant_min = quant_min or 0
        quant_max = quant_max or 15
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, range_shrink_percentile=range_shrink_percentile, **kwargs)


####################################################################
class AdaptiveLowBITPower2PerChannelWeightObserver(AdaptivePower2PerChannelWeightObserver):
    def __init__(self, *args, quant_min=None, quant_max=None, **kwargs):
        quant_min = quant_min or -8
        quant_max = quant_max or +7
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, **kwargs)


class AdaptiveLowBITPower2ActivationObserver(AdaptivePower2ActivationObserver):
    def __init__(self, *args, quant_min=None, quant_max=None, range_shrink_percentile=observer_utils.RANGE_SHRINK_PERCENTILE_LOWBIT, **kwargs):
        quant_min = quant_min or 0
        quant_max = quant_max or 15
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, range_shrink_percentile=range_shrink_percentile, **kwargs)


####################################################################
class AdaptiveRangeRestricted4LowBITPerChannelWeightObserver(AdaptiveLowBITPerChannelWeightObserver):
    def __init__(self, *args, range_val=4.0, **kwargs) -> None:
        super().__init__(*args, range_val=range_val, **kwargs)


class AdaptiveRangeRestricted4LowBITActivationObserver(AdaptiveLowBITActivationObserver):
    def __init__(self, *args, range_val=4.0, **kwargs) -> None:
        super().__init__(*args, range_val=range_val, **kwargs)


####################################################################
ADAPTIVE_WEIGHT_OBSERVER_TYPES = (AdaptiveWeightObserver,
                                  AdaptivePerChannelWeightObserver,
                                  AdaptivePower2WeightObserver,
                                  AdaptivePower2PerChannelWeightObserver,
                                  AdaptiveLowBITPerChannelWeightObserver,
                                  AdaptiveLowBITPower2PerChannelWeightObserver,
                                  AdaptiveRangeRestricted4LowBITPerChannelWeightObserver)

ADAPTIVE_ACTIVATION_OBSERVER_TYPES = (AdaptiveActivationObserver,
                                      AdaptivePower2ActivationObserver,
                                      AdaptiveLowBITActivationObserver,
                                      AdaptiveLowBITPower2ActivationObserver,
                                      AdaptiveRangeRestricted4LowBITActivationObserver)

ADAPTIVE_OBSERVER_TYPES = tuple(list(ADAPTIVE_WEIGHT_OBSERVER_TYPES) + list(ADAPTIVE_ACTIVATION_OBSERVER_TYPES))
