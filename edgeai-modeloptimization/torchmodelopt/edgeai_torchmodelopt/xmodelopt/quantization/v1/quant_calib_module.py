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

###########################################################
# Approximate quantized floating point simulation with gradients.
# Can be used for quantized training of models.
###########################################################

import torch
import numpy as np
import copy
import warnings

from ....xnn import layers
from ....xnn import utils
from .quant_train_module import *

warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)


###########################################################
class QuantCalibrateModule(QuantTrainModule):
    def __init__(self, module, dummy_input, *args, bitwidth_weights=8, bitwidth_activations=8, per_channel_q=False,
                 histogram_range=True, bias_calibration=True, constrain_weights=None,
                 range_shrink_weights=None, range_shrink_activations=None,
                 power2_weight_range=None, power2_activation_range=None, constrain_bias=None, lr_calib=0.05,
                 quantize_in=True, quantize_out=True, verbose_mode=False, **kwargs):
        self.weights_calibration = False
        self.lr_calib = lr_calib
        self.calibration_factor = lr_calib
        self.calibration_gamma = 0.5
        self.calibrate_repeats = 1
        self.quantize_enable = True
        self.update_activation_range = True
        constrain_weights = (bias_calibration and (not per_channel_q)) if constrain_weights is None else constrain_weights
        super().__init__(module, dummy_input, *args, bitwidth_weights=bitwidth_weights, bitwidth_activations=bitwidth_activations,
                         per_channel_q=per_channel_q, histogram_range=histogram_range, bias_calibration=bias_calibration,
                         constrain_weights=constrain_weights, constrain_bias=constrain_bias,
                         range_shrink_weights=range_shrink_weights, range_shrink_activations=range_shrink_activations,
                         power2_weight_range=power2_weight_range, power2_activation_range=power2_activation_range,
                         quantize_in=quantize_in, quantize_out=quantize_out, verbose_mode=verbose_mode, **kwargs)
        self.calib_stats = dict()


    def forward(self, inputs, *args, **kwargs):
        # calibration doesn't need gradients
        with torch.no_grad():
            # counters such as num_batches_tracked are used. update them.
            self.update_counters()
            # bitwidth_warmup
            self.adjust_bitwidth()

            # backup the current state
            training = self.training

            # Set all bns to eval so that they do not change. We can't set the whole model to eval because,
            # we need the pact to learn the ranges - which will happen only in training mode.
            # Also the model output itself may be different in eval mode (in certain cases -
            # for example if in a segmentation model argmax is done instead of softmax in eval mode).
            utils.freeze_bn(self)

            # actual forward call
            if self.training and (self.bias_calibration or self.weights_calibration):
                # calibration
                outputs = self.forward_calibrate(inputs, *args, **kwargs)
            else:
                outputs = self.module(inputs, *args, **kwargs)
            #

            self.train(training)
        #
        return outputs


    def forward_calibrate(self, inputs, *args, **kwargs):
        # we don't need gradients for calibration
        # prepare/backup weights
        if self.num_batches_tracked == 0:
            # lr_calib
            self.calibration_factor = self.lr_calib * np.power(self.calibration_gamma, float(self.epoch))
            # backup original weights
            self._backup_weights_orig()
            # backup quantized weights
            self._backup_weights_quant()
        #

        # Compute the mean output in float first.
        outputs = self.forward_float(inputs, *args, **kwargs)
        # Then adjust weights/bias so that the quantized output matches float output
        outputs = self.forward_quantized(inputs, *args, **kwargs)
        # not needed outside - clear
        self.calib_stats = dict()
        return outputs


    def forward_float(self, inputs, *args, **kwargs):
        self._restore_weights_orig()
        # disable quantization for a moment
        quantize_enable_backup_value, update_activation_range_backup_value = self.quantize_enable, self.update_activation_range
        utils.apply_setattr(self, quantize_enable=False, update_activation_range=False)

        self.add_call_hook(self.module, self.forward_float_hook)
        outputs = self.module(inputs, *args, **kwargs)
        self.remove_call_hook(self.module)

        # turn quantization back on - not a clean method
        utils.apply_setattr(self, quantize_enable=quantize_enable_backup_value, update_activation_range=update_activation_range_backup_value)
        self._backup_weights_orig()
        return outputs
    #
    def forward_float_hook(self, op, *inputs_orig):
        outputs = op.__forward_orig__(*inputs_orig)

        # calibration at specific layers
        output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        while isinstance(output, (list, tuple)):
            output = output[0]

        reduce_dims = [0, 2, 3] if len(output.size()) == 4 else ([0] if len(output.size()) == 2 else None)

        bias = op.bias if hasattr(op, 'bias') else None
        output_mean = output_std = None
        if (self.bias_calibration and bias is not None):
            output_mean = torch.mean(output, dim=reduce_dims).data
        #

        if self.weights_calibration and utils.is_conv_deconv(op):
            output_std = torch.std(output, dim=reduce_dims).data
        #
        self.calib_stats[op] = dict(mean=output_mean, std=output_std)
        return outputs
    #


    def forward_quantized(self, input, *args, **kwargs):
        self._restore_weights_quant()
        self.add_call_hook(self.module, self.forward_quantized_hook)
        for _ in range(self.calibrate_repeats):
            output = self.module(input, *args, **kwargs)
        #
        self.remove_call_hook(self.module)
        self._backup_weights_quant()
        return output
    #
    def forward_quantized_hook(self, op, input, *args, **kwargs):
        outputs = op.__forward_orig__(input, *args, **kwargs)

        # calibration at specific layers
        output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        while isinstance(output, (list, tuple)):
            output = output[0]

        bias = op.bias if hasattr(op, 'bias') else None
        output_mean_float = self.calib_stats[op]['mean']
        output_std_float = self.calib_stats[op]['std']
        if self.bias_calibration and bias is not None:
            reduce_dims = [0,2,3] if len(output.size()) == 4 else ([0] if len(output.size()) == 2 else None)
            output_mean = torch.mean(output, dim=reduce_dims).data
            output_delta = output_mean_float - output_mean
            output_delta = output_delta * self.calibration_factor
            bias.data += (output_delta)
        #

        if self.weights_calibration and utils.is_conv_deconv(op):
            eps = 1e-6
            weight = op.weight
            reduce_dims = [0, 2, 3] if len(output.size()) == 4 else ([0] if len(output.size()) == 2 else None)
            output_std = torch.std(output, dim=reduce_dims).data
            output_ratio = (output_std_float + eps) / (output_std + eps)
            channels = output.size(1)
            output_ratio = output_ratio.view(channels, 1, 1, 1) if len(weight.data.size()) > 1 else output_ratio
            output_ratio = torch.pow(output_ratio, self.calibration_factor)
            weight.data *= output_ratio
        #
        return outputs

