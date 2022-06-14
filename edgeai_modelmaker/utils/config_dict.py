#################################################################################
# Copyright (c) 2018-2022, Texas Instruments Incorporated - http://www.ti.com
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


# a simple config node class
class AttrDict(dict):
    def __init__(self):
        super().__init__()
        self.__dict__ = self

    def merge_from(self, src_cfg):
        if src_cfg is not None:
            for src_key, src_val in src_cfg.items():
                self[src_key] = src_val
            #
        #
        return self

    def clone(self):
        new_cfg = type(self)()
        new_cfg.merge_from(self)
        return new_cfg

    def __deepcopy__(self, memodict):
        new_config = self.clone()
        memodict[id(self)] = new_config
        return new_config


# ConfigDict is derived from AttrDict
class ConfigDict(AttrDict):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, (dict, ConfigDict)):
                self.update(**arg)
            #
        #
        for k, v in kwargs.items():
            if k in self:
                if isinstance(self[k], (dict, ConfigDict)) and isinstance(v, (dict, ConfigDict)):
                    self[k].update(v)
                else:
                    self[k] = v
                #
            elif isinstance(v, dict):
                self[k] = ConfigDict(v)
            else:
                self[k] = v
            #
        #
        return self
