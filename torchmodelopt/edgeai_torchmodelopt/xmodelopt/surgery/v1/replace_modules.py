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

import copy
import functools
import torch
import inspect
import math


__all__ = ['replace_modules']


def replace_modules(model, inplace=True, replacement_dict=None, **kwargs):
    assert replacement_dict is not None, 'replacement_dict must be provided'
    model = model if inplace else copy.deepcopy(inplace)
    for p_name, parent_m in model.named_modules():
        for c_name, current_m in parent_m.named_children():
            if not _replace_with_new_module(parent_m, c_name, current_m, replacement_dict, **kwargs):
                replace_modules(current_m, inplace=inplace, replacement_dict=replacement_dict, **kwargs)
            #
        #
    #
    return model


def _replace_with_new_module(parent, c_name, current_m, replacement_dict, **kwargs):
    for k_check, v_params in replacement_dict.items():
        assert callable(k_check), f'the key in replacement_dict must be a class or function: {k_check}'
        if inspect.isclass(k_check):
            do_replace = isinstance(current_m, k_check)
        else:
            do_replace = k_check(current_m)
        #
        if do_replace:
            # first entry is the constructor or a callable that constructs
            new_constructor = v_params[0] if isinstance(v_params, (list,tuple)) else v_params
            assert callable(new_constructor), f'the value in replacement_dict must be a class or function: {new_constructor}'
            # the parameters of the new moulde that has to be copied from current
            new_args = {}
            if isinstance(v_params, (list,tuple)) and len(v_params) > 1:
                for v_params_k in v_params[1:]:
                    if isinstance(v_params_k, dict):
                        new_args.update(v_params_k)
                    elif isinstance(v_params_k, str):
                        new_args.update({v_params_k:getattr(current_m,v_params_k)})
                    #
                #
            #
            # create the new module that replaces the existing
            if inspect.isclass(new_constructor):
                new_m = new_constructor(**new_args, **kwargs)
            else:
                new_m = new_constructor(current_m, **new_args, **kwargs)
            #
            # now initialize the new module and replace it in the parent
            if new_m is not current_m:
                if not hasattr(new_m, 'is_initialized') or new_m.is_initialized is False:
                    _initialize_module(new_m)
                new_m.train(current_m.training)
                # requires_grad setting of the source is used for the newly created module
                requires_grad = None
                for param_cur in current_m.parameters():
                    requires_grad = requires_grad or param_cur.requires_grad
                #
                if requires_grad is not None:
                    for param_new in new_m.parameters():
                        param_new.requires_grad = requires_grad
                    #
                #
                setattr(parent, c_name, new_m)
                return True
            #
        #
    #
    return False


def _initialize_module(module):
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Linear):
            init_range = 1.0 / math.sqrt(m.out_features)
            torch.nn.init.uniform_(m.weight, -init_range, init_range)
            torch.nn.init.zeros_(m.bias)
        #
    #
