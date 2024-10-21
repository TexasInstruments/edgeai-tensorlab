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

import torch
from .quant_base import QuantFxBaseModule
import warnings


class QATFxModule(QuantFxBaseModule):
    def __init__(self, *args, backend='qnnpack', is_qat=True, **kwargs):
        warnings.warn("Fx based quantization wrapper will be depercated in the future after pt2e quantization wrapper is completed.")
        super().__init__(*args, is_qat=is_qat, backend=backend, **kwargs)


class PTCFxModule(QuantFxBaseModule):
    '''
    Post Training Calibration (PTC) for Quantization is similar to Post Training Quantization
    PTC can be integrated inton the training script easily with couple of lines of change. 
    It is faster than QAT as it doesn't actively train the weights.
    '''
    def __init__(self, *args, backend='qnnpack', is_qat=False, bias_calibration_factor=0.01, num_batch_norm_update_epochs=0, num_observer_update_epochs=1, **kwargs):
        # warnings.warn("Fx based quantization wrapper will be depercated in the future after pt2e quantization wrapper is completed.")
        super().__init__(*args, is_qat=is_qat, backend=backend, bias_calibration_factor=bias_calibration_factor, \
            num_batch_norm_update_epochs=num_batch_norm_update_epochs, num_observer_update_epochs=num_observer_update_epochs, **kwargs)