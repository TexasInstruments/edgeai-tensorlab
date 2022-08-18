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


__all__ = ['prepare', 'load_weights', 'freeze', 'train', 'eval', 'convert']


def _set_quant_backend(backend=None):
    if backend:
        if backend not in torch.backends.quantized.supported_engines:
            raise RuntimeError("Quantized backend not supported: " + str(backend))
        #
        torch.backends.quantized.engine = backend
    #


def load_weights(model, pretrained=None, change_names_dict=None):
    # Load weights for accuracy evaluation of a QAT model
    if pretrained is not None and pretrained is not False:
        print("=> using pre-trained model from {}".format(pretrained))
        if hasattr(model, 'load_weights'):
            model.load_weights(pretrained, download_root='./data/downloads', change_names_dict=change_names_dict)
        else:
            xnn.utils.load_weights(model, pretrained, download_root='./data/downloads', change_names_dict=change_names_dict)
        #
    #


def prepare(model, qconfig_dict=None, pretrained=None, pretrained_after_prepare=False, backend=None,
            num_batch_norm_update_epochs=None, num_observer_update_epochs=None, prepare_fn=None):
    _set_quant_backend(backend=backend)
    if qconfig_dict is None:
        qconfig_dict = {"": torch.quantization.get_default_qat_qconfig(backend)}
    #
    model.train()
    if not pretrained_after_prepare:
        load_weights(model, pretrained=pretrained)
    #
    model = prepare_fn(model, qconfig_dict)
    if pretrained_after_prepare:
        load_weights(model, pretrained=pretrained)
    #
    # fake quantization for qat
    model.apply(torch.ao.quantization.enable_fake_quant)
    # observes for range estimation
    model.apply(torch.ao.quantization.enable_observer)
    # store additional information
    model.__quant_info__ = dict(num_batch_norm_update_epochs=num_batch_norm_update_epochs,
                                 num_observer_update_epochs=num_observer_update_epochs,
                                 num_epochs_tracked=0)
    return model


def freeze(model, freeze_bn=True, freeze_observers=True):
    if freeze_observers is True:
        model.apply(torch.ao.quantization.disable_observer)
    elif freeze_observers is False:
        model.apply(torch.ao.quantization.enable_observer)
    #
    if freeze_bn is True:
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    elif freeze_bn is False:
        model.apply(torch.nn.intrinsic.qat.update_bn_stats)
    #
    return model


def unfreeze(model, unfreeze_bn=True, unfreeze_observers=True):
    freeze(model, not unfreeze_bn, not unfreeze_observers)
    return model


def _get_quant_info(model):
    for m in model.modules():
        if hasattr(m, '__quant_info__'):
            return m.__quant_info__
        #
    #
    return None


def train(model):
    # put the model in train mode
    model.train()
    # freezing ranges after a few epochs improve accuracy
    __quant_info__ = _get_quant_info(model)
    if __quant_info__ is not None:
        num_batch_norm_update_epochs = __quant_info__['num_batch_norm_update_epochs']
        num_observer_update_epochs = __quant_info__['num_observer_update_epochs']
        num_epochs_tracked = __quant_info__['num_epochs_tracked']
        __quant_info__['num_epochs_tracked'] += 1
    else:
        num_batch_norm_update_epochs = None
        num_observer_update_epochs = None
        num_epochs_tracked = 0
    #
    num_batch_norm_update_epochs = num_batch_norm_update_epochs or 4
    num_observer_update_epochs = num_observer_update_epochs or 6
    freeze(model, freeze_bn=(num_epochs_tracked>=num_batch_norm_update_epochs),
           freeze_observers=(num_epochs_tracked>=num_observer_update_epochs))
    return model


def eval(model):
    model.eval()
    freeze(model, freeze_bn=True, freeze_observers=True)
    return model


def convert(model, convert_fn=quantize_fx.convert_fx):
    # make a copy inorder not to alter the original
    model = copy.deepcopy(model)
    # convert requires cpu model
    model = model.to(torch.device('cpu'))
    # now do the actual conversion
    model = convert_fn(model)
    return model



