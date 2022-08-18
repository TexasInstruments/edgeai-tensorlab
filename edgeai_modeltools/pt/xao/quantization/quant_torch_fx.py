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

import copy
import torch
import torch.quantization.quantize_fx as quantize_fx
from ... import xnn
from . import _quant_torch_base as quant_torch_base

__all__ = ['prepare', 'load_weights', 'freeze', 'train', 'eval', 'convert']


def load_weights(model, *args, **kwargs):
    return quant_torch_base.load_weights(model, *args, **kwargs)


def prepare(model, *args, prepare_fn=quantize_fx.prepare_qat_fx, is_qat=True, **kwargs):
    model = quant_torch_base.prepare(model, *args, prepare_fn=prepare_fn, **kwargs)
    return model


def freeze(model):
    model = quant_torch_base.freeze(model)
    return model


def unfreeze(model):
    model = quant_torch_base.unfreeze(model)
    return model


def train(model):
    model = quant_torch_base.train(model)
    return model


def eval(model):
    model = quant_torch_base.eval(model)
    return model


def convert(model, convert_fn=quantize_fx.convert_fx, inplace=False):
    model = quant_torch_base.convert(model, convert_fn=convert_fn, inplace=inplace)
    return model


##################################################################
# this is a convenient Module form of the above APIs
class QuantTorchFxModule(torch.nn.Module):
    def __int__(self, module, qconfig_dict=None, pretrained=None,
            pretrained_after_prepare=False, backend=None,
            num_batch_norm_update_epochs=None, num_observer_update_epochs=None):
        super().__init__()
        self.module = prepare(module, qconfig_dict=qconfig_dict, pretrained=pretrained,
            pretrained_after_prepare=pretrained_after_prepare, backend=backend,
            num_batch_norm_update_epochs=num_batch_norm_update_epochs,
            num_observer_update_epochs=num_observer_update_epochs)

    def load_weights(self, pretrained):
        load_weights(self.module, pretrained=pretrained)

    def train(self):
        self.module = train(self.module)

    def eval(self):
        self.module = eval(self.module)

    def freeze(self):
        self.module = freeze(self.module)

    def unfreeze(self):
        self.module = freeze(self.module)

    def convert(self):
        return convert(self.module)
