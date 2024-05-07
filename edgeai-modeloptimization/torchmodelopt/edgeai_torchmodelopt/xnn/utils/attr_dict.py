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


# config node is derived from AttrDict
# important node: This class handles a list specially - it will be split when the split when split() is called
# this is to support multiple decoder in a multi-task network.
# so use a tuple when specifiying array-like params for one decoder and encapsulate the tuple in a list for multi-task.
class ConfigNode(AttrDict):
    def __init__(self):
        super().__init__()

    def split(self, index):
        new_config = self.clone()
        for src_key, src_val in self.items():
            if isinstance(src_val, list):
                assert index < len(src_val), 'Model_config parameter {} is a list. If its a list, the length {} \
                                                is expected to match the number of decoders.'.format(src_key, len(src_val))
                new_config[src_key] = src_val[index]
            else:
                new_config[src_key] = src_val
        #
        return new_config