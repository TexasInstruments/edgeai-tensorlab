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
import torch.nn as nn
from . import quant_pt2e_func


class QuantPT2EBaseModule(nn.Module):
    def __init__(self, model, *args, quantizer=None, **kwargs):
        '''
        model: input model to be used for inserting quantization
        '''
        super().__init__()
        self.module = quant_pt2e_func.init(model, quantizer=quantizer, *args, **kwargs)
        
    def load_weights(self, *args, **kwargs):
        quant_pt2e_func.load_weights(self.module, *args, **kwargs)

    def train(self, *args, **kwargs):
        return quant_pt2e_func.train(self.module, *args, **kwargs)
    
    def calibrate(self, *args, **kwargs):
        return quant_pt2e_func.calibrate(self.module, *args, **kwargs)

    def freeze(self, *args, **kwargs):
        return quant_pt2e_func.freeze(self.module, *args, **kwargs)

    def unfreeze(self, *args, **kwargs):
        return quant_pt2e_func.unfreeze(self.module, *args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def convert(self, *args, **kwargs):
        self.module = quant_pt2e_func.convert(self.module, *args, **kwargs)
        return self

    def export(self, *args, **kwargs):
        return quant_pt2e_func.export(self.module, *args, **kwargs)
        
            
    