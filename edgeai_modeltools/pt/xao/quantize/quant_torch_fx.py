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


__all__ = ['prepare', 'freeze', 'convert', 'train', 'eval']


def set_quant_backend(backend=None):
    if backend:
        if backend not in torch.backends.quantized.supported_engines:
            raise RuntimeError("Quantized backend not supported: " + str(backend))
        #
        torch.backends.quantized.engine = backend
    #


def load_weights(module, pretrained=None, change_names_dict=None):
    # Load weights for accuracy evaluation of a QAT model
    if pretrained is not None and pretrained is not False:
        print("=> using pre-trained model from {}".format(pretrained))
        if hasattr(module, 'load_weights'):
            module.load_weights(pretrained, download_root='./data/downloads', change_names_dict=change_names_dict)
        else:
            xnn.utils.load_weights(module, pretrained, download_root='./data/downloads', change_names_dict=change_names_dict)
        #
    #


def prepare(module, qconfig_dict=None, inplace=False, pretrained=None, pretrained_after_prepare=False, backend=None):
    set_quant_backend(backend=backend)
    if qconfig_dict is None:
        qconfig_dict = {"": torch.quantization.get_default_qat_qconfig(backend)}
    #
    module.train()
    if not pretrained_after_prepare:
        load_weights(module, pretrained=pretrained)
    #
    module = quantize_fx.prepare_qat_fx(module, qconfig_dict)
    if pretrained_after_prepare:
        load_weights(module, pretrained=pretrained)
    #
    # fake quantization for qat
    module.apply(torch.ao.quantization.enable_fake_quant)
    # observes for range estimation
    module.apply(torch.ao.quantization.enable_observer)
    return module


def freeze(module, freeze_bn=True, freeze_observers=True):
    if freeze_observers is True:
        module.apply(torch.ao.quantization.disable_observer)
    elif freeze_observers is False:
        module.apply(torch.ao.quantization.enable_observer)
    #
    if freeze_bn is True:
        module.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    elif freeze_bn is False:
        module.apply(torch.nn.intrinsic.qat.update_bn_stats)
    #
    return module


def unfreeze(module, unfreeze_bn=True, unfreeze_observers=True):
    freeze(module, not unfreeze_bn, not unfreeze_observers)
    return module


def train(module):
    module.train()
    unfreeze(module, unfreeze_bn=True, unfreeze_observers=True)
    return module


def eval(module):
    module.eval()
    freeze(module, freeze_bn=True, freeze_observers=True)
    return module


def convert(module, inplace=False):
    module = quantize_fx.convert_fx(module)
    return module
