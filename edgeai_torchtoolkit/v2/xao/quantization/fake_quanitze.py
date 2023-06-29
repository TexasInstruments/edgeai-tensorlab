import math
import torch
from torch.ao.quantization import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver, FakeQuantize
from ....v1 import xnn


# FusedMovingAvgObsFakeQuantize will not use calculate_qparams() only during convert
# it directly calls torch.fused_moving_avg_obs_fake_quant() which implements everything inside it
# so use FakeQuantize here as we need to override calculate_qparams()

####################################################################
class AdaptiveWeightFakeQuantize(FakeQuantize):
    '''
    Create a subclass, just to distinguish between the ones used for activation and weight
    '''
    pass


class AdaptiveActivationFakeQuantize(FakeQuantize):
    '''
    Create a subclass, just to distinguish between the ones used for activation and weight
    '''
    pass


####################################################################
class AdaptiveWeightNoQuantize(FakeQuantize):
    '''
    Create a subclass, just to distinguish between the ones used for activation and weight
    '''
    def forward(self, X):
        no_quantize = True
        if no_quantize:
            # just clamp during warmup - this dummy forward is just to collect stats
            self.disable_fake_quant()
            super().forward(X.detach())
            self.enable_fake_quant()
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
            return torch.clamp(X, min_val.clone(), max_val.clone())
        else:
            return super().forward(X)
        #


class AdaptiveActivationNoQuantize(FakeQuantize):
    '''
    Create a subclass, just to distinguish between the ones used for activation and weight
    '''
    def forward(self, X):
        no_quantize = True
        if no_quantize:
            # just clamp during warmup - this dummy forward is just to collect stats
            self.disable_fake_quant()
            super().forward(X.detach())
            self.enable_fake_quant()
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
            return torch.clamp(X, min_val.clone(), max_val.clone())
        else:
            return super().forward(X)
        #


####################################################################
ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES = (AdaptiveWeightFakeQuantize,
                                    AdaptiveWeightNoQuantize)

ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES = (AdaptiveActivationFakeQuantize,
                                        AdaptiveActivationNoQuantize)

ADAPTIVE_FAKE_QUANT_TYPES = tuple(list(ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES) + list(ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES))
