import math
import torch
from torch.ao.quantization import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver, FakeQuantize
from ....v1 import xnn


# FusedMovingAvgObsFakeQuantize will not use calculate_qparams() only during convert
# it directly calls torch.fused_moving_avg_obs_fake_quant() which implements everything inside it
# so use FakeQuantize here as we need to override calculate_qparams()

class AdaptiveFakeQuantize(FakeQuantize):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detect_change = False
        self.smooth_change = True
        self.delta_change = 0.0
        self.delta_binned_history = torch.zeros((1,))
        self.history_available = False
        self.momentum = 0.9
        self.compact_mode = False
        self.compact_bins = 32

    def set_adaptive_params(self, detect_change=None, compact_mode=None, smooth_change=None):
        if detect_change is not None:
            self.detect_change = detect_change
        #
        if compact_mode is not None:
            self.compact_mode = compact_mode
        #
        if smooth_change is not None:
            self.smooth_change = smooth_change
        #

    def forward(self, X):
        x_q = super().forward(X)
        if self.training and self.detect_change:
            delta = (x_q - X).detach()
            if self.compact_mode:
                delta_binned = torch.histc(delta, bins=self.compact_bins) / delta.numel()
                if self.history_available:
                    delta_change = torch.sum(torch.abs(delta_binned - self.delta_binned_history).float()).item() / 2.0
                #
            else:
                delta_binned = torch.sign(delta)
                if self.history_available:
                    delta_change = torch.mean((delta_binned != self.delta_binned_history).float()).item()
                #
            #
            if self.history_available:
                if self.smooth_change:
                    delta_change = delta_change * (1-self.momentum) + self.delta_change * self.momentum
                #
                self.delta_change = delta_change
            #
            self.delta_binned_history = delta_binned
            self.history_available = True
        #
        return x_q


####################################################################
class AdaptiveWeightFakeQuantize(AdaptiveFakeQuantize):
    '''
    Create a subclass, just to distinguish between the ones used for activation and weight
    '''
    pass


class AdaptiveActivationFakeQuantize(AdaptiveFakeQuantize):
    '''
    Create a subclass, just to distinguish between the ones used for activation and weight
    '''
    pass


####################################################################
ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES = (AdaptiveWeightFakeQuantize,)

ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES = (AdaptiveActivationFakeQuantize,)

ADAPTIVE_FAKE_QUANT_TYPES = tuple(list(ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES) + list(ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES))
