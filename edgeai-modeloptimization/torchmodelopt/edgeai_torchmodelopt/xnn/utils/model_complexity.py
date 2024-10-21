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
import torchinfo


def get_model_complexity(model, *inp, verbose = None):
    try:
        result = torchinfo.summary(model, input_data=inp, verbose=verbose)
        total_mult_adds = result.total_mult_adds
        total_params = result.total_params
    except:
        print("torchinfo failed to get_model_complexity.")
        # print("torchinfo failed to get_model_complexity. trying out alternative implementation - may be approximate")
        # result = _get_model_complexity(model, *inp)
        # total_mult_adds = result.total_mult_adds
        # total_params = result.total_params
        total_mult_adds = 0
        total_params = 0
    #
    return total_mult_adds, total_params


def _get_model_complexity_forward_hook_func(module, inp, oup):
    mult_adds = 0
    param_count = sum([p.numel() for p in module.parameters(recurse=False) if p.requires_grad])
    if isinstance(module, torch.nn.Conv2d):
        oup_volume = oup.shape[0] * oup.shape[1] * oup.shape[2] * oup.shape[3]
        per_pixel_macs = module.weight.shape[0] * module.weight.shape[1] * module.weight.shape[2] * module.weight.shape[3]
        mult_adds = oup_volume * per_pixel_macs
    elif isinstance(module, torch.nn.Linear):
        oup_volume = oup.shape[0] * oup.shape[1] * oup.shape[2] * oup.shape[3]
        per_pixel_macs = module.weight.shape[0] * module.weight.shape[1] * module.weight.shape[2] * module.weight.shape[3]
        mult_adds = oup_volume * per_pixel_macs
    #
    module.__model_complexity_mult_adds__ += mult_adds
    module.__model_complexity_params__ = param_count
    return


def _get_model_complexity(model, *inp):
    # add a hook function
    for module_name, module in model.named_modules():
        module.__model_complexity_hook_func__ = module.register_forward_hook(_get_model_complexity_forward_hook_func)
        module.__model_complexity_mult_adds__ = 0
        module.__model_complexity_params__ = 0
    #
    # actual forward
    _ = model(*inp)
    # now add up the gmacs and params
    result = object()
    result.total_mult_adds = 0
    result.total_params = 0
    for module_name, module in model.named_modules():
        if hasattr(module, '__model_complexity_mult_adds__'):
            result.total_mult_adds += module.__model_complexity_mult_adds__
            del module.__model_complexity_mult_adds__
        #
        if hasattr(module, '__model_complexity_params__'):
            result.total_params += module.__model_complexity_params__
            del module.__model_complexity_params__
        #
        if hasattr(module, '__model_complexity_hook_func__'):
            del module.__model_complexity_hook_func__
        #
    #
    return result