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

def _get_last_bias_module_sequential(module):
    last_m = None
    for m_idx, m in enumerate(list(module)[::-1]):
        if isinstance(m, torch.nn.Sequential):
            return _get_last_bias_module_sequential(m)
        elif hasattr(m, 'bias') and m.bias is not None:
            return m
        #
    return last_m


def get_last_bias_modules(module):
    last_ms = []
    if hasattr(module, 'conv') and hasattr(module, 'res'):
        last_ms.append(_get_last_bias_module_sequential(module.conv))
        if hasattr(module, 'res') and module.res is not None:
            last_ms.append(_get_last_bias_module_sequential(module.res))
        #
    elif isinstance(module, torch.nn.Sequential):
        last_ms.append(_get_last_bias_module_sequential(module))
    elif hasattr(module, 'bias') and module.bias is not None:
        last_ms.append(module)
    else:
        for m in list(module.modules())[::-1]:
            if hasattr(m, 'bias') and m.bias is not None:
                last_ms.append(m)
                break
            #
        #
    #
    last_ms = [m for m in last_ms if m is not None]
    return last_ms

