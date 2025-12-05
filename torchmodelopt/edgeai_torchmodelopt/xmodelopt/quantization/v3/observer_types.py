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
import warnings
import torch
import torch.ao.quantization

from .... import xnn
from . import observer_utils
from . import quant_utils


####################################################################
# select which histogram observer to use in this file as the base

# histogram observers derived from pytorch's observers. cumulative one is sutable for activations.
# observer_utils.MSEHistogramObserverBase
# observer_utils.MovingAverageMSEHistogramObserverBase

# fast histogram observers defined by us. cumulative one is sutable for activations.
# these range shrink ones may help in the case of CNNs
# observer_utils.RangeShrinkHistogramObserverBase
# observer_utils.MovingAverageRangeShrinkHistogramObserverBase


####################################################################
class AdaptiveWeightObserver(torch.ao.quantization.MinMaxObserver):
    def __init__(self, *args, quant_min=-128, quant_max=+127, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, power2_scale=False, 
                 range_max=None, fixed_range=False, **kwargs):
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, dtype=dtype, qscheme=qscheme, **kwargs)
        self.symmetric = (qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric))
        self.power2_scale = power2_scale
        self.range_max = range_max
        self.fixed_range = fixed_range
        self.freeze_observer = False

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            #
        #
        
    def get_min_max(self) -> tuple[torch.Tensor, torch.Tensor, bool]:
        return observer_utils._correct_min_max(self.min_val, self.max_val)

    def _check_min_max_valid(self, min_val: torch.Tensor, max_val: torch.Tensor) -> bool:
        return observer_utils._check_min_max_valid(min_val, max_val)
    
    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        scale, zero_point = observer_utils._calculate_qparams(super()._calculate_qparams, 
                min_val, max_val, self.quant_min, self.quant_max, self.symmetric, self.power2_scale, self.eps)
        return scale, zero_point

    def forward(self, x_orig):
        if not torch.is_floating_point(x_orig):
            return x_orig
        #
        if self.freeze_observer:
            return x_orig
        #
        x = x_orig.detach()
        x = torch.where(torch.isfinite(x), x, observer_utils.eps)
        x = super().forward(x)
        if self.range_max is not None:
            signed_range = torch.min(self.min_val.detach()).item() < 0.0
            min_val = (-self.range_max) if signed_range else 0.0
            max_val = (+self.range_max) if signed_range else (+self.range_max)
            if self.fixed_range:
                self.min_val.fill_(min_val)
                self.max_val.fill_(max_val)
            else:
                self.min_val = torch.clamp(self.min_val, min=min_val, max=0.0)
                self.max_val = torch.clamp(self.max_val, min=0.0, max=max_val)
            #
        #
        return x_orig


class AdaptivePerChannelWeightObserver(torch.ao.quantization.PerChannelMinMaxObserver):
    def __init__(self, *args, quant_min=-128, quant_max=+127, dtype=torch.qint8, qscheme=torch.per_channel_symmetric, power2_scale=False, 
                 range_max=None, fixed_range=False, **kwargs):
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, dtype=dtype, qscheme=qscheme, **kwargs)
        self.symmetric = (qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric))
        self.power2_scale = power2_scale
        self.range_max = range_max
        self.fixed_range = fixed_range
        self.freeze_observer = False

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            #
        #

    def get_min_max(self) -> tuple[torch.Tensor, torch.Tensor, bool]:
        return observer_utils._correct_min_max(self.min_val, self.max_val)

    def _check_min_max_valid(self, min_val: torch.Tensor, max_val: torch.Tensor) -> bool:
        return observer_utils._check_min_max_valid(min_val, max_val)
    
    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        # weights qparams are always symmetric and this is ensured inside the super class, no need to handle it here.
        scale, zero_point = observer_utils._calculate_qparams(super()._calculate_qparams, 
                min_val, max_val, self.quant_min, self.quant_max, self.symmetric, self.power2_scale, self.eps)
        return scale, zero_point

    def forward(self, x_orig):
        if not torch.is_floating_point(x_orig):
            return x_orig
        #
        if self.freeze_observer:
            return x_orig
        #
        x = x_orig.detach()
        x = torch.where(torch.isfinite(x), x, observer_utils.eps)
        x = super().forward(x)
        if self.range_max is not None:
            signed_range = torch.min(self.min_val.detach()).item() < 0.0
            min_val = (-self.range_max) if signed_range else 0.0
            max_val = (+self.range_max) if signed_range else (+self.range_max)
            if self.fixed_range:
                self.min_val.fill_(min_val)
                self.max_val.fill_(max_val)
            else:
                self.min_val = torch.clamp(self.min_val, min=min_val, max=0.0)
                self.max_val = torch.clamp(self.max_val, min=0.0, max=max_val)
            #
        #
        return x_orig


####################################################################
class AdaptiveMinMaxActivationObserver(torch.ao.quantization.MinMaxObserver):
    def __init__(self, *args, quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, power2_scale=False, 
                 range_max=None, fixed_range=False, range_shrink_percentile=0, **kwargs):
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, dtype=dtype, qscheme=qscheme, **kwargs)
		# activation quantization cannot use torch.per_channel_symmetric, it has to be torch.per_tensor_symmetric
        self.symmetric = (qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric))
        self.power2_scale = power2_scale
        self.range_max = range_max
        self.fixed_range = fixed_range
        self.freeze_observer = False
        self.num_batches_tracked = 0

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            #
        #

    def get_min_max(self) -> tuple[torch.Tensor, torch.Tensor, bool]:
        return observer_utils._correct_min_max(self.min_val, self.max_val)

    def _check_min_max_valid(self, min_val: torch.Tensor, max_val: torch.Tensor) -> bool:
        return observer_utils._check_min_max_valid(min_val, max_val)
    
    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        scale, zero_point = observer_utils._calculate_qparams(super()._calculate_qparams, 
                min_val, max_val, self.quant_min, self.quant_max, self.symmetric, self.power2_scale, self.eps)
        return scale, zero_point

    def forward(self, x_orig):
        if not torch.is_floating_point(x_orig):
            return x_orig
        #
        if self.freeze_observer:
            return x_orig
        #
        x = x_orig.detach()
        x = torch.where(torch.isfinite(x), x, observer_utils.eps)
        x = super().forward(x)
        if self.range_max is not None:
            signed_range = torch.min(self.min_val.detach()).item() < 0.0
            min_val = (-self.range_max) if signed_range else 0.0
            max_val = (+self.range_max) if signed_range else (+self.range_max)
            if self.fixed_range:
                self.min_val.fill_(min_val)
                self.max_val.fill_(max_val)
            else:
                self.min_val = torch.clamp(self.min_val, min=min_val, max=0.0)
                self.max_val = torch.clamp(self.max_val, min=0.0, max=max_val)
            #
        #
        return x_orig


class AdaptiveMovingAverageMinMaxActivationObserver(torch.ao.quantization.MovingAverageMinMaxObserver):
    def __init__(self, *args, quant_min=0, quant_max=255, dtype=torch.quint8, qscheme=torch.per_tensor_affine, power2_scale=False, 
                 range_max=None, fixed_range=False, range_shrink_percentile=0, **kwargs):
        super().__init__(*args, quant_min=quant_min, quant_max=quant_max, dtype=dtype, qscheme=qscheme, **kwargs)
		# activation quantization cannot use torch.per_channel_symmetric, it has to be torch.per_tensor_symmetric
        self.symmetric = (qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric))
        self.power2_scale = power2_scale
        self.range_max = range_max
        self.fixed_range = fixed_range
        self.freeze_observer = False
        self.num_batches_tracked = 0

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            #
        #

    def get_min_max(self) -> tuple[torch.Tensor, torch.Tensor, bool]:
        return observer_utils._correct_min_max(self.min_val, self.max_val)

    def _check_min_max_valid(self, min_val: torch.Tensor, max_val: torch.Tensor) -> bool:
        return observer_utils._check_min_max_valid(min_val, max_val)
    
    @torch.jit.export
    def _calculate_qparams(self, min_val, max_val):
        r"""Calculates the quantization parameters."""
        scale, zero_point = observer_utils._calculate_qparams(super()._calculate_qparams, 
                min_val, max_val, self.quant_min, self.quant_max, self.symmetric, self.power2_scale, self.eps)
        return scale, zero_point

    def forward(self, x_orig):
        if not torch.is_floating_point(x_orig):
            return x_orig
        #
        if self.freeze_observer:
            return x_orig
        #
        x = x_orig.detach()
        x = torch.where(torch.isfinite(x), x, observer_utils.eps)
        x = super().forward(x)
        if self.range_max is not None:
            signed_range = torch.min(self.min_val.detach()).item() < 0.0
            min_val = (-self.range_max) if signed_range else 0.0
            max_val = (+self.range_max) if signed_range else (+self.range_max)
            if self.fixed_range:
                self.min_val.fill_(min_val)
                self.max_val.fill_(max_val)
            else:
                self.min_val = torch.clamp(self.min_val, min=min_val, max=0.0)
                self.max_val = torch.clamp(self.max_val, min=0.0, max=max_val)
            #
        #
        return x_orig


class AdaptiveActivationObserver(AdaptiveMovingAverageMinMaxActivationObserver):
    pass


####################################################################
class _AdaptiveOutlierSuppressionObserver(observer_utils.AdaptiveRangeShrinkObserver):
    def __init__(self, *args, range_shrink=True, **kwargs):
        range_shrink = observer_utils.RangeShrinkPercentileValues.AGGRESSIVE if isinstance(range_shrink, (bool,)) else range_shrink
        super().__init__(*args, range_shrink=range_shrink, **kwargs)

    @torch.jit.export
    def _calculate_qparams(
        self, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.ones_like(min_val)
        zero_point = torch.zeros_like(min_val, dtype=torch.int64)
        return scale, zero_point


class AdaptiveOutlierSuppressionWeightObserver(_AdaptiveOutlierSuppressionObserver):
    pass


class AdaptiveOutlierSuppressionActivationObserver(_AdaptiveOutlierSuppressionObserver):
    pass


####################################################################
ADAPTIVE_WEIGHT_OBSERVER_TYPES = (AdaptiveWeightObserver,
                                  AdaptivePerChannelWeightObserver,
                                  AdaptiveOutlierSuppressionWeightObserver)

ADAPTIVE_ACTIVATION_OBSERVER_TYPES = (AdaptiveActivationObserver, 
                                      AdaptiveMinMaxActivationObserver, 
                                      AdaptiveMovingAverageMinMaxActivationObserver,
                                      AdaptiveOutlierSuppressionActivationObserver)

ADAPTIVE_OBSERVER_TYPES = tuple(list(ADAPTIVE_WEIGHT_OBSERVER_TYPES) + list(ADAPTIVE_ACTIVATION_OBSERVER_TYPES))


# ####################################################################
# # example custom weight observers
# AdaptivePower2WeightObserver = xnn.utils.partialclass(AdaptiveWeightObserver, power2_scale=True, class_name='AdaptivePower2WeightObserver')
# AdaptivePerChannelPower2WeightObserver = xnn.utils.partialclass(AdaptivePerChannelWeightObserver, power2_scale=True, class_name='AdaptivePerChannelPower2WeightObserver')
# AdaptivePerChannelFixedRange4WeightObserver = xnn.utils.partialclass(AdaptivePerChannelWeightObserver, range_max=4.0, fixed_range=True, class_name='AdaptivePerChannelFixedRange4WeightObserver')
#
# AdaptivePerChannelBit4WeightObserver = xnn.utils.partialclass(AdaptivePerChannelWeightObserver, quant_min=-8, quant_max=7, range_shrink_percentile=observer_utils.RANGE_SHRINK_PERCENTILE_AGGRESSIVE, class_name='AdaptivePerChannelBit4WeightObserver')
# AdaptivePerChannelBit4MaxRange4WeightObserver = xnn.utils.partialclass(AdaptivePerChannelWeightObserver, quant_min=-8, quant_max=7, range_max=4.0, range_shrink_percentile=observer_utils.RANGE_SHRINK_PERCENTILE_AGGRESSIVE, class_name='AdaptivePerChannelBit4MaxRange4WeightObserver')
# AdaptivePerChannelBit4FixedRange4WeightObserver = xnn.utils.partialclass(AdaptivePerChannelWeightObserver, quant_min=-8, quant_max=7, range_max=4.0, fixed_range=True, range_shrink_percentile=observer_utils.RANGE_SHRINK_PERCENTILE_AGGRESSIVE, class_name='AdaptivePerChannelBit4FixedRange4WeightObserver')
#
# # example custom activation observers
# AdaptiveSymActivationObserver = xnn.utils.partialclass(AdaptiveActivationObserver, qscheme=torch.per_tensor_symmetric, class_name='AdaptiveSymActivationObserver')
# AdaptiveSymPower2ActivationObserver = xnn.utils.partialclass(AdaptiveActivationObserver, qscheme=torch.per_tensor_symmetric, power2_scale=True, class_name='AdaptiveSymPower2ActivationObserver')
# AdaptiveSymPower2FixedRange4ActivationObserver = xnn.utils.partialclass(AdaptiveActivationObserver, qscheme=torch.per_tensor_symmetric, power2_scale=True, range_max=4, fixed_range=True, class_name='AdaptiveSymPower2FixedRange4ActivationObserver')
# AdaptiveFixedRange4ActivationObserver = xnn.utils.partialclass(AdaptiveActivationObserver, range_max=4.0, fixed_range=True, class_name='AdaptiveFixedRange4ActivationObserver')
#
# AdaptiveBit4ActivationObserver = xnn.utils.partialclass(AdaptiveActivationObserver, quant_min=0, quant_max=15, range_shrink_percentile=observer_utils.RANGE_SHRINK_PERCENTILE_AGGRESSIVE, class_name='AdaptiveBit4ActivationObserver')
# AdaptiveBit4MaxRange4ActivationObserver = xnn.utils.partialclass(AdaptiveActivationObserver, quant_min=0, quant_max=15, range_max=4.0, range_shrink_percentile=observer_utils.RANGE_SHRINK_PERCENTILE_AGGRESSIVE, class_name='AdaptiveBit4MaxRange4ActivationObserver')
# AdaptiveBit4FixedRange4ActivationObserver = xnn.utils.partialclass(AdaptiveActivationObserver, quant_min=0, quant_max=15, range_max=4.0, fixed_range=True, range_shrink_percentile=observer_utils.RANGE_SHRINK_PERCENTILE_AGGRESSIVE, class_name='AdaptiveBit4FixedRange4ActivationObserver')
