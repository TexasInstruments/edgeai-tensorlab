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

import copy
import warnings

from .quant_graph_module import *


###########################################################
# default settings for quantization

class ConstrainBiasType:
    CONSTRAIN_BIAS_TYPE_NONE = 0
    CONSTRAIN_BIAS_TYPE_SATURATE = 1
    CONSTRAIN_BIAS_TYPE_REDUCE_WEIGHT_SCALE = 2
    CONSTRAIN_BIAS_TYPE_REDUCE_FEATURE_SCALE = 3


class QuantDefaults:
    RANGE_SHRINK_WEIGHTS_DEFAULT = 0.0
    POWER2_WEIGHT_RANGE_DEFAULT = True
    POWER2_ACTIVATION_RANGE_DEFAULT = True
    CONSTRAIN_BIAS_DEFAULT = ConstrainBiasType.CONSTRAIN_BIAS_TYPE_SATURATE


###########################################################
# base module to be use for all quantization modules
class QuantBaseModule(QuantGraphModule):
    def __init__(self, module, dummy_input, *args, bitwidth_weights=8, bitwidth_activations=8, per_channel_q=False,
                 histogram_range=True, bias_calibration=False, constrain_weights=False, constrain_bias=None,
                 range_shrink_weights=None, range_shrink_activations=None,
                 power2_weight_range=None, power2_activation_range=None, model_surgery_quantize=True, 
                 quantize_in=True, quantize_out=True, verbose_mode=False, **kwargs):
        
        warnings.warn("WARNING - xnn.quantization is based on modules. For latest functionality, please use the torch.fx based xmodelopt.quantization instead")

        super().__init__(module, quantize_in=quantize_in, quantize_out=quantize_out)
        self.bitwidth_weights = bitwidth_weights
        self.bitwidth_activations = bitwidth_activations
        self.per_channel_q = per_channel_q
        self.histogram_range = histogram_range
        self.constrain_weights = constrain_weights
        self.bias_calibration = bias_calibration
        self.verbose_mode = verbose_mode

        self.power2_weight_range = power2_weight_range if (power2_weight_range is not None) else \
            QuantDefaults.POWER2_WEIGHT_RANGE_DEFAULT
        self.power2_activation_range = power2_activation_range if (power2_activation_range is not None) else \
            QuantDefaults.POWER2_ACTIVATION_RANGE_DEFAULT
        # range shrink - 0.0 indicates no shrink
        self.range_shrink_weights = range_shrink_weights if (range_shrink_weights is not None) else \
            QuantDefaults.RANGE_SHRINK_WEIGHTS_DEFAULT
        self.range_shrink_activations = range_shrink_activations if (range_shrink_activations is not None) else \
            layers.PAct2.PACT2_RANGE_SHRINK_DEFAULT
        # constrain_bias means bias that is being added to accumulator is limited to 16bit (for 8bit quantization).
        # scale factor to be used for constrain_bias is the product of scale factors of weight and input
        self.constrain_bias = constrain_bias if (constrain_bias is not None) else \
            QuantDefaults.CONSTRAIN_BIAS_DEFAULT
        if self.per_channel_q and self.constrain_bias == ConstrainBiasType.CONSTRAIN_BIAS_TYPE_SATURATE:
            warnings.warn('Per channel quantization can increase the weight scale a lot, resulting in a lot of \
                bias saturation if constrain_bias is enabled. Too much bias saturation can hurt accuracy. \
                Suggest to reduce weight scale by passing constrain_bias as CONSTRAIN_BIAS_TYPE_REDUCE_WEIGHT_SCALE \
                to avoid bias saturation.')
        #

        # using per_channel_q when constrain_bias is set may not be good for accuracy.
        if self.constrain_bias and self.per_channel_q:
            warnings.warn('using per_channel_q when constrain_bias is set may not be good for accuracy.')
        #

        if (self.per_channel_q == 'all'):
            assert self.constrain_bias == ConstrainBiasType.CONSTRAIN_BIAS_TYPE_NONE, \
                f'constrain_bias must be {ConstrainBiasType.CONSTRAIN_BIAS_TYPE_NONE} \
                when per_channel_q is all. Got {self.constrain_bias}'
        #
        
        # for help in debug/print
        utils.add_module_names(self)
        # put in eval mode before analyze
        self.eval()
        # model surgery for quantization
        if model_surgery_quantize:
            with torch.no_grad():
                utils.print_yellow("=> model surgery by '{}'".format(self.model_surgery_quantize.__name__))
                assert dummy_input is not None, 'dummy input is needed by quantized models to analyze graph'
                self.model_surgery_quantize(dummy_input, *args, **kwargs)
            #
            # add hooks to execute the pact modules
            self.add_activation_hooks()
        #
        # for help in debug/print
        utils.add_module_names(self)

        # set attributes to all modules - can control the behaviour from here
        utils.apply_setattr(self, bitwidth_weights=self.bitwidth_weights, bitwidth_activations=self.bitwidth_activations,
                            histogram_range=histogram_range, bias_calibration=self.bias_calibration, per_channel_q=self.per_channel_q,
                            range_shrink_weights=self.range_shrink_weights, range_shrink_activations=self.range_shrink_activations,
                            power2_weight_range=self.power2_weight_range, power2_activation_range=self.power2_activation_range,
                            constrain_weights=self.constrain_weights, constrain_bias=self.constrain_bias, verbose_mode=self.verbose_mode)

    def add_activation_hooks(self):
        for m in self.modules():
            if hasattr(m, 'activation_in'):
                m.register_forward_pre_hook(self._forward_input_activation)
            #
            if hasattr(m, 'activation_q'):
                m.register_forward_hook(self._forward_output_activation)
            #
        #

    # add a forward hook to call the extra activation that we added
    def _forward_input_activation(self, op, inputs):
        # hook passes the input as tuple - expand it
        to_squeeze = isinstance(inputs, tuple) and len(inputs) == 1
        inputs = inputs[0] if to_squeeze else inputs
        inputs = op.activation_in(inputs)
        inputs = (inputs,) if to_squeeze else inputs
        return inputs

    def _forward_output_activation(self, op, inputs, outputs):
        # hook passes the input as tuple - expand it
        to_squeeze = isinstance(outputs, tuple) and len(outputs) == 1
        outputs = outputs[0] if to_squeeze else outputs
        outputs = op.activation_q(outputs)
        outputs = (outputs,) if to_squeeze else outputs
        return outputs

    def apply_setattr(self, **kwargs):
        utils.apply_setattr(self, **kwargs)


    def train(self, mode=True):
        self.iter_in_epoch = -1
        super().train(mode)


    def _backup_weights_orig(self):
        self.__params_orig__ = {}
        for n,p in self.named_parameters():
            self.__params_orig__[n] = copy.deepcopy(p.data)
        #
        self.__buffers_orig__ = {}
        for n,p in self.named_buffers():
            self.__buffers_orig__[n] = copy.deepcopy(p.data)
        #

    def _restore_weights_orig(self):
        for n,p in self.named_parameters():
            p.data.copy_(self.__params_orig__[n].data)
        #
        for n,p in self.named_buffers():
            p.data.copy_(self.__buffers_orig__[n].data)
        #

    def _backup_weights_quant(self):
        self.__params_quant__ = {}
        for n,p in self.named_parameters():
            self.__params_quant__[n] = copy.deepcopy(p.data)
        #
        self.__buffers_quant__ = {}
        for n,p in self.named_buffers():
            self.__buffers_quant__[n] = copy.deepcopy(p.data)
        #

    def _restore_weights_quant(self):
        for n,p in self.named_parameters():
            p.data.copy_(self.__params_quant__[n].data)
        #
        for n,p in self.named_buffers():
            p.data.copy_(self.__buffers_quant__[n].data)
        #

    def _remove_backups(self):
        if hasattr(self, '__params_orig__'):
            del self.__params_orig__
        if hasattr(self, '__params_quant__'):
            del self.__params_quant__
        if hasattr(self, '__buffers_orig__'):
            del self.__params_orig__
        if hasattr(self, '__buffers_quant__'):
            del self.__params_quant__
        #
        # output means are some temp buffers used for calibration
        def _remove_output_means_op(self, op):
            if hasattr(op, '__output_mean_orig__'):
                del op.__output_mean_orig__
            if hasattr(op, '__output_std_orig__'):
                del op.__output_std_orig__
            #
        #
        self.apply(_remove_output_means_op)






