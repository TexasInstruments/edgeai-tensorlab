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

def module_weights_init(module, a=0, nonlinearity='relu', weight_init='normal', mode='fan_out', bias_init=0.0):
    assert nonlinearity in ('relu', 'leaky_relu'), f"nonlinearity must be one of ('relu', 'leaky_relu'), got: {nonlinearity}"
    # weight initialization
    for m in module.modules():
        # not ConvTranspose2d is not handled here.
        # let pytorch's default initialization be used for now.
        if isinstance(m, torch.nn.Conv2d):
            if weight_init == 'normal':
                torch.nn.init.kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            elif weight_init == 'uniform':
                torch.nn.init.kaiming_uniform_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            else:
                assert False, f"unknown init type. must be one of ('normal', 'uniform'), got: {weight_init}"
            #
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, bias_init)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
            torch.nn.init.ones_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, bias_init)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            torch.nn.init.constant_(m.bias, bias_init)




