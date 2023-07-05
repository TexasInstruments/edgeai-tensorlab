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
        self.adaptive_factor = 0.0

    def set_adaptive_factor(self, adaptive_factor):
        self.adaptive_factor = adaptive_factor

    def forward(self, X):
        x_q = super().forward(X)
        if self.adaptive_factor != 0.0:
            min_val, max_val = self.activation_post_process.min_val, self.activation_post_process.max_val
            if min_val.ndim > 0:
                if X.ndim == 2:
                    min_val = min_val.unsqueeze(-1)
                    max_val = max_val.unsqueeze(-1)
                elif X.ndim == 3:
                    min_val = min_val.unsqueeze(-1).unsqueeze(-1)
                    max_val = max_val.unsqueeze(-1).unsqueeze(-1)
                elif X.ndim == 4:
                    min_val = min_val.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    max_val = max_val.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                #
            #
            x_noq = torch.clamp(X, min_val.clone(), max_val.clone())
            x_q = x_noq * self.adaptive_factor + x_q * (1.0-self.adaptive_factor)
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
