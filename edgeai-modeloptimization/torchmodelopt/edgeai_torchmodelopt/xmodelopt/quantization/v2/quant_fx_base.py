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
from . import quant_fx_func
from .quant_fx_func import ModelQuantFormat


class QuantFxBaseModule(torch.nn.Module):
    def __init__(self, model, *args, passthrough_attributes=None, **kwargs):
        '''
        model: input model to be used for QAT
        qconfig_type: qconfig_type can be one of the modes defined in qconfig_types (string)
            or it can be a dict that will be passed to qconfig_types.get_config_from_dict()
            it can also be an instance of torch.ao.quantization.QConfig as used when using torch.ao.quantization apis
        '''
        super().__init__()
        self.module = quant_fx_func.init(model, *args, add_methods=True, **kwargs)
        self._add_passthrough_attributes(passthrough_attributes)

    def load_weights(self, *args, **kwargs):
        quant_fx_func.load_weights(self.module, *args, **kwargs)

    def train(self, *args, **kwargs):
        return quant_fx_func.train(self.module, *args, **kwargs)

    def freeze(self, *args, **kwargs):
        return quant_fx_func.freeze(self.module, *args, **kwargs)

    def unfreeze(self, *args, **kwargs):
        return quant_fx_func.unfreeze(self.module, *args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def convert(self, *args, **kwargs):
        return quant_fx_func.convert(self.module, *args, **kwargs)

    def export(self, *args, **kwargs):
        return quant_fx_func.export(self.module, *args, **kwargs)

    def _add_passthrough_attributes(self, passthrough_attributes):
        if passthrough_attributes is not None:
            for attribute_name in passthrough_attributes:
                if hasattr(self.module, attribute_name):
                    attribute_getter = lambda self: getattr(self.module, attribute_name)
                    attribute_setter = lambda self, value: setattr(self.module, attribute_name, value)
                    new_property = property(fget=attribute_getter, fset=attribute_setter)
                    setattr(self.__class__, attribute_name, new_property)
                #
            #
