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

import types
import warnings
import torch


class HookedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # default method name that is replaced up is 'forward' - but multi-gpu training fails with that
    # but if we replace __call__ everything seems okay.
    def add_call_hook(self, module, function_new, method_name='forward', backup_name='__forward_orig__'):
        def _call_hook_enable(op):
            # do not patch the top level modules. makes it easy to invoke by self.module(x)
            if op is not module:
                assert not hasattr(op, backup_name), f'in {op.__class__.__name__} detected an existing function {backup_name} : please double check'
                # backup the original forward of op into backup_name
                method_orig = getattr(op, method_name)
                setattr(op, backup_name, method_orig)
                # set new method
                method_new = types.MethodType(function_new, op)
                setattr(op, method_name, method_new)
            #
        #
        # apply to all children
        module.apply(_call_hook_enable)


    def remove_call_hook(self, module, method_name='forward', backup_name='__forward_orig__'):
        def _call_hook_disable(op):
            # do not patch the top level modules. makes it easy to invoke by self.module(x)
            if op is not module:
                if hasattr(op, backup_name):
                    method_new = getattr(op, method_name)
                    # restore the original forward method which is now stored as backup
                    method_orig = getattr(op, backup_name)
                    setattr(op, method_name, method_orig)
                    # delete the backup
                    setattr(op, backup_name, method_new)
                    delattr(op, backup_name)
                #
            #
        #
        # apply to all children
        module.apply(_call_hook_disable)

