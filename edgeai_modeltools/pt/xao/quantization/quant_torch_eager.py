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
import torch.quantization as quantization
from ... import xnn
from . import _quant_torch_base as quant_torch_base


__all__ = ['prepare', 'load_weights', 'freeze', 'train', 'eval', 'convert']


def load_weights(model, *args, **kwargs):
    return quant_torch_base.load_weights(model, *args, **kwargs)


def _get_fuse_list(self, module, dummy_input):
    for name, m in module.named_modules():
        m.__track_modules_name__ = name
    #
    def _track_modules1(m, inp, oup):
        prev_module = inp.__track_modules_m__[-1] if hasattr(inp, '__track_modules_m__') else None
        if prev_module is not None:
            if hasattr(prev_module, '__track_modules_next__'):
                prev_module.__track_modules_next__.append(m)
            else:
                prev_module.__track_modules_next__ = [m]
            #
            if hasattr(m, '__track_modules_prev__'):
                m.__track_modules_prev__.append(prev_module)
            else:
                m.__track_modules_prev__ = [prev_module]
            #
        #
        if hasattr(oup, '__track_modules_m__'):
            oup.__track_modules_m__.append(m)
        else:
            oup.__track_modules_m__ = [m]
        #
    #
    def _track_modules(m, inp, oup):
        inp = inp if isinstance(inp, (list,tuple)) else [inp]
        oup = inp if isinstance(oup, (list,tuple)) else [oup]
        for input in inp:
            for output in oup:
                _track_modules1(m, input, output)
            #
        #
    #
    for m in module.modules():
        m.__track_modules_m_hook__ = m.register_forward_hook(_track_modules)
    #
    module(dummy_input)
    # analyze
    fuse_list = []
    for m in module.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            m_next = None
            m_next2 = None
            if hasattr(m, '__track_modules_next__') and len(m.__track_modules_next__) == 1:
                m_next = m.__track_modules_next__[-1]
                if hasattr(m_next, '__track_modules_next__') and len(m_next.__track_modules_next__) == 1:
                    m_next2 = m_next.__track_modules_next__[-1]
                #
            #
            if isinstance(m_next, torch.nn.BatchNorm2d) and isinstance(m_next2, (torch.nn.ReLU,torch.nn.ReLU6)):
                fuse_list.append([m.__track_modules_name__, m_next.__track_modules_name__, m_next2.__track_modules_name__])
            elif isinstance(m_next, torch.nn.BatchNorm2d):
                fuse_list.append([m.__track_modules_name__, m_next.__track_modules_name__])
            elif isinstance(m_next, (torch.nn.ReLU,torch.nn.ReLU6)):
                fuse_list.append([m.__track_modules_name__, m_next.__track_modules_name__])
            #
        # elif isinstance(m, layers.FloatFunctionalBlock):
        #     if isinstance(m_next, (torch.nn.ReLU,torch.nn.ReLU6)):
        #         fuse_list.append([m.__track_modules_name__, m_next.__track_modules_name__])
        #     #
        # #
    #
    for m in module.modules():
        if hasattr(m, '__track_modules_m_hook__'):
            m.__track_modules_m_hook__.remove()
            del m.__track_modules_m_hook__
        #
        if hasattr(m, '__track_modules_m__'):
            del m.__track_modules_m__
        #
        if hasattr(m, '__track_modules_prev__'):
            del m.__track_modules_prev__
        #
        if hasattr(m, '__track_modules_next__'):
            del m.__track_modules_next__
        #
        if hasattr(m, '__track_modules_name__'):
            del m.__track_modules_name__
        #
    #
    return fuse_list


def prepare(model, *args, dummy_input=None, prepare_fn=quantization.prepare_qat, inplace=False, **kwargs):
    if hasattr(model, 'fuse_model'):
        model.fuse_model()
    else:
        device = next(self.module.parameters()).device
        dummy_input = dummy_input.to(device=device)
        fuse_list = _get_fuse_list(model, dummy_input)
        model = torch.ao.quantization.fuse_modules(model, fuse_list, inplace=inplace)
    #
    for p in model.modules():
        for n, m in p.named_children():
            if isinstance(m, xnn.layers.AddBlock):
                setattr(p, n, xnn.layers.FloatFunctionalBlock('add'))
            elif isinstance(m, xnn.layers.MultBlock):
                setattr(p, n, xnn.layers.FloatFunctionalBlock('mul'))
            elif isinstance(m, xnn.layers.CatBlock):
                setattr(p, n, xnn.layers.FloatFunctionalBlock('cat'))
            #
        #
    #
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


def convert(model, convert_fn=quantization.convert):
    model = quant_torch_base.convert(model, convert_fn=convert_fn)
    return model


##################################################################
# this is a convenient Module form of the above APIs
class QuantTorchEagerModule(torch.nn.Module):
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
