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

from .quant_train_module import *

class QuantTestModule(QuantTrainModule):
    def __init__(self, module, dummy_input, *args, bitwidth_weights=8, bitwidth_activations=8, per_channel_q=False,
                 histogram_range=True, bias_calibration=False, constrain_weights=False, model_surgery_quantize=True,
                 range_shrink_weights=None, range_shrink_activations=None,
                 power2_weight_range=None, power2_activation_range=None, constrain_bias=None, 
                 quantize_in=True, quantize_out=True, verbose_mode=False, **kwargs):
        super().__init__(module, dummy_input, *args, bitwidth_weights=bitwidth_weights, bitwidth_activations=bitwidth_activations,
                         per_channel_q=per_channel_q, histogram_range=histogram_range, bias_calibration=bias_calibration,
                         constrain_weights=constrain_weights, constrain_bias=constrain_bias,
                         range_shrink_weights=range_shrink_weights, range_shrink_activations=range_shrink_activations,
                         power2_weight_range=power2_weight_range, power2_activation_range=power2_activation_range,
                         quantize_in=quantize_in, quantize_out=quantize_out, verbose_mode=verbose_mode, **kwargs)
        assert model_surgery_quantize == True, f'{self.__class__.__name__} does not support model_surgery_quantize=False. please use a qat or calibrated module.'
        self.eval()


    def train(self, mode=True):
        assert mode == False, 'QuantTestModule cannot be used in train mode'
        super().train(mode)

