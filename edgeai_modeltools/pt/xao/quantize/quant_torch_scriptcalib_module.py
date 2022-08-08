#################################################################################
# Copyright (c) 2018-2021, Texas Instruments Incorporated - http://www.ti.com
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
from .quant_torch_qconfig import *
from .quant_torch_base_module import *

#warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)


###########################################################
class QuantTorchScriptCalibrateModule(QuantTorchBaseModule):
    def __init__(self, module, dummy_input, *args, with_fakequantize=False, **kwargs):
        '''Quantize after after exporting the model to torch script.
        TODO: For some reason get_custom_qconfig_with_fakequantize fails in inference after convert for this scriptcalib model
        '''
        super().__init__(module, dummy_input, *args,  **kwargs)
        ts_module = torch.jit.trace(module, dummy_input)
        self.module = ts_module
        self.with_fakequantize = with_fakequantize


    def fuse_model(self, inplace=True):
        return self.module


    def prepare(self):
        super().prepare()
        qconfig_func = get_custom_qconfig_with_fakequantize if self.with_fakequantize else get_custom_qconfig
        qconfig_args = dict(histogram=self.histogram, symmetric=self.symmetric, per_channel=self.per_channel_q,
                  power2_weight_range=self.power2_weight_range, power2_activation_range=self.power2_activation_range)
        qconfig = qconfig_func(**qconfig_args)
        self.module = torch.quantization.prepare_jit(self.module, qconfig_dict={'':qconfig}, inplace=True)
        if self.per_channel_q_depthwise_only:
            self._force_per_channel_depthwise_only(qconfig_func, qconfig_args)
        #

    def convert(self):
        super().convert()
        self.module = torch.quantization.convert_jit(self.module, inplace=True)

    def export(self, *args, **kwargs):
        return self.module

