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
        self.history_tensor1 = torch.zeros((1,))
        self.history_tensor2 = torch.zeros((1,))
        self.momentum = 0.9
        self.histogram_mode = False
        self.histogram_bins = 256
        self.num_batches_tracked = 0

    def set_adaptive_params(self, detect_change=None, histogram_mode=None, smooth_change=None):
        if detect_change is not None:
            self.detect_change = detect_change
        #
        if histogram_mode is not None:
            self.histogram_mode = histogram_mode
        #
        if smooth_change is not None:
            self.smooth_change = smooth_change
        #

    def forward(self, X):
        x_q = super().forward(X)
        with torch.no_grad():
            if self.training and self.detect_change:
                if self.histogram_mode:
                    diff0 = (x_q - X)
                    if self.num_batches_tracked > 0:
                        diff1 = self.history_tensor1
                        min_val = torch.minimum(diff0.min(), diff1.min())
                        max_val = torch.maximum(diff0.max(), diff1.max())
                        delta_change0 = self.compute_histogram_delta(diff0, min_val=min_val, max_val=max_val)
                        delta_change1 = self.compute_histogram_delta(diff1, min_val=min_val, max_val=max_val)
                        delta_change = torch.abs(delta_change0 - delta_change1)
                        delta_change = delta_change.sum().item() #/ 2.0
                        if self.smooth_change and self.num_batches_tracked > 1:
                            delta_change = delta_change * (1-self.momentum) + self.delta_change * self.momentum
                        #
                        self.delta_change = delta_change
                    #
                    self.history_tensor1 = diff0.data
                else:
                    delta_change0 = torch.sign(x_q - X)
                    if self.num_batches_tracked > 0:
                        delta_change1 = self.history_tensor1
                        delta_diff = (delta_change0 != delta_change1)
                        delta_change = torch.mean(delta_diff.float()).item()
                        if self.smooth_change and self.num_batches_tracked > 1:
                            delta_change = delta_change * (1-self.momentum) + self.delta_change * self.momentum
                        #
                        self.delta_change = delta_change
                    #
                    self.history_tensor1 = delta_change0.data
                #
                self.num_batches_tracked += 1
            #
        #
        return x_q

    def compute_histogram_delta(self, tensor_input, min_val=0, max_val=0):
        delta_change = torch.histc(tensor_input, bins=self.histogram_bins, min=min_val, max=max_val) / tensor_input.numel()
        return delta_change


####################################################################
class AdaptiveWeightFakeQuantize(AdaptiveFakeQuantize):
    '''
    Create a subclass, just to distinguish between the ones used for activation and weight
    '''
    def forward(self, X):
        # to preserve sparsity in the weights
        sparsity_mask = (X != 0).detach()
        X = X * sparsity_mask
        # this is the actual fake_quntize
        x_q = super().forward(X)
        return x_q


class AdaptiveActivationFakeQuantize(AdaptiveFakeQuantize):
    '''
    Create a subclass, just to distinguish between the ones used for activation and weight
    '''
    pass


####################################################################
ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES = (AdaptiveWeightFakeQuantize,)

ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES = (AdaptiveActivationFakeQuantize,)

ADAPTIVE_FAKE_QUANT_TYPES = tuple(list(ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES) + list(ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES))
