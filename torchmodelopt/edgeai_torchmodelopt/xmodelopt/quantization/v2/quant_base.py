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
from . import quant_func


class QuantFxBaseModule(torch.nn.Module):
    def __init__(self, model, *args, add_methods=True, **kwargs):
        '''
        model: input model to be used for QAT / PTC
        qconfig_type: qconfig_type can be one of the modes defined in qconfig_types (string)
            or it can be a dict that will be passed to qconfig_types.get_config_from_dict()
            it can also be an instance of torch.ao.quantization.QConfig as used when using torch.ao.quantization apis
        '''
        super().__init__()
        self.module = quant_func.init(model, *args, add_methods=add_methods, **kwargs)

    def load_weights(self, *args, **kwargs):
        quant_func.load_weights(self.module, *args, **kwargs)

    def train(self, *args, **kwargs):
        return quant_func.train(self.module, *args, **kwargs)
    
    def calibrate(self, *args, **kwargs):
        return quant_func.calibrate(self.module, *args, **kwargs)

    def freeze(self, *args, **kwargs):
        return quant_func.freeze(self.module, *args, **kwargs)

    def unfreeze(self, *args, **kwargs):
        return quant_func.unfreeze(self.module, *args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def convert(self, *args, **kwargs):
        self.module = quant_func.convert(self.module, *args, **kwargs)
        return self

    def export(self, *args, **kwargs):
        return quant_func.export(self.module, *args, **kwargs)

