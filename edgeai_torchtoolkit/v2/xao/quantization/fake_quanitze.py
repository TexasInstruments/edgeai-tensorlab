import math
import torch
from torch.ao.quantization import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver, FakeQuantize
from ....v1 import xnn


class AdaptiveWeightFakeQuantize(FakeQuantize):
    '''
    Create a subclass, just to distinguish between the ones used for activation and weight
    '''
    def __int__(self, *args, **kwargs):
        self.warmup_flag = False

    def set_warmup_flag(self, value):
        self.warmup_flag = value

    def forward(self, X):
        if self.warmup_flag:
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


class AdaptiveActivationFakeQuantize(FakeQuantize):
    '''
    Create a subclass, just to distinguish between the ones used for activation and weight
    '''
    def __int__(self, *args, **kwargs):
        self.warmup_flag = False

    def set_warmup_flag(self, value):
        self.warmup_flag = value

    def forward(self, X):
        if self.warmup_flag:
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
ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES = (AdaptiveWeightFakeQuantize,)

ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES = (AdaptiveActivationFakeQuantize,)

ADAPTIVE_FAKE_QUANT_TYPES = tuple(list(ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES) + list(ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES))
