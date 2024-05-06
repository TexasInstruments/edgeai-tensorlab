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
import collections
import torch
from .. import layers


def is_normalization(module):
    is_norm = isinstance(module, (torch.nn.BatchNorm2d, layers.DefaultNorm2d,
                                 torch.nn.GroupNorm, layers.GroupBatchNorm2d))
    return is_norm


def is_activation(module):
    is_act = isinstance(module, (torch.nn.ReLU, torch.nn.ReLU6, torch.nn.Hardtanh,
                                 layers.PAct2, layers.QAct, layers.NoQAct))
    return is_act

def is_pact2(module):
    is_act = isinstance(module, (layers.PAct2))
    return is_act

def is_conv(module):
    return isinstance(module, torch.nn.Conv2d)

def is_deconv(module):
    return isinstance(module, torch.nn.ConvTranspose2d)

def is_conv_deconv(module):
    return isinstance(module, (torch.nn.Conv2d,torch.nn.ConvTranspose2d))

def is_conv_deconv_linear(module):
    return isinstance(module, (torch.nn.Conv2d,torch.nn.ConvTranspose2d,torch.nn.Linear))

def is_linear(module):
    return isinstance(module, torch.nn.Linear)

def is_dwconv(module):
    return is_conv(module) and (module.weight.size(1) == 1)


def is_bn(module):
    return isinstance(module, torch.nn.BatchNorm2d)


def get_parent_module(module, m):
    for p_name, p in module.named_modules():
        for c_name, c in p.named_children():
            if c is m:
                return p
    #
    return None



def get_module_name(module, m):
    if module is None:
        return None
    #
    for name, mod in module.named_modules():
        if mod is m:
            return name
    #
    return None


def is_tensor(inp):
    return isinstance(inp, torch.Tensor)


def is_none(inp):
    if is_tensor(inp):
        return False
    else:
        return (inp is None)


def is_not_none(inp):
    return not is_none(inp)


def is_list(inp):
    return isinstance(inp, (list, tuple))


def is_not_list(inp):
    return not is_list(inp)


def is_fixed_range(op):
    return isinstance(op, (torch.nn.ReLU6, torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.Hardtanh, \
                           layers.PAct2))


def get_range(op):
    if isinstance(op, layers.PAct2):
        return op.get_clips_act()
    elif isinstance(op, torch.nn.ReLU6):
        return 0.0, 6.0
    elif isinstance(op, torch.nn.Sigmoid):
        return 0.0, 1.0
    elif isinstance(op, torch.nn.Tanh):
        return -1.0, 1.0
    elif isinstance(op, torch.nn.Hardtanh):
        return op.min_val, op.max_val
    else:
        assert False, 'dont know the range of the module'


def add_module_names(model):
    for n, m in model.named_modules():
        m.name = n
    #
    return model


def squeeze_list(inputs):
    return inputs[0] if (is_list(inputs) and len(inputs)==1) else inputs


def squeeze_list2(inputs):
    return inputs[0] if (is_list(inputs) and (len(inputs)==1) and is_list(inputs[0])) else inputs


def make_list(inputs):
    return inputs if is_list(inputs) else (inputs,)


def apply_setattr(model, always=False, **kwargs):
    assert len(kwargs) >= 1, 'atlest one keyword argument must be specified. ..=.., in addition always=.. can be specified.'
    def setattr_func(op):
        for name, value in kwargs.items():
            if hasattr(op, name) or always:
                setattr(op, name, value)
            #
        #
    #
    model.apply(setattr_func)


def clear_grad(model):
    for p in model.parameters():
        p.grad = None
    #


class ModelAverage():
    def __init__(self, maxlen=12):
        self.history = collections.deque(maxlen=maxlen)

    def put(self, m):
        self.history.append([copy.deepcopy(p) for p in m.parameters()])
        return m

    def get(self, m):
        history_len = len(self.history)
        if history_len < 1:
            return m
        #
        m_sum = self.history[0]
        for idx in range(1, history_len):
            mi = self.history[idx]
            m_sum = [p0.data+p1.data for p0, p1 in zip(m_sum, mi)]
        #
        m_mean = [ms/history_len for ms in m_sum]

        m_copy = copy.deepcopy(m)
        for p, p_mean in zip(m_copy.parameters(), m_mean):
            p.data.copy_(p_mean.data)

        return m_copy
