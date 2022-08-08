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

import copy
import torch

from .quant_torch_qconfig import *
from .quant_torch_base_module import *

#warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)

###########################################################
class QuantTorchEagerCalibrateModule(QuantTorchBaseModule):
    def __init__(self, module, dummy_input, *args, histogram=True, with_fakequantize=True, **kwargs):
        '''
        Quantize after collecting the ranges.
        If with_fakequantize is set, the range collection will be with quantization
        of weights and activations - this is seen to help accuracy.
        '''
        super().__init__(module, dummy_input, *args,  histogram=histogram, with_fakequantize=with_fakequantize,
                         constrain_weights=False, **kwargs)


