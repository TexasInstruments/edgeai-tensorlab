# Copyright (c) 2018-2021, Texas Instruments
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

import os
import yaml
from . import misc_utils


class ConfigDict(dict):
    def __init__(self, input=None, **kwargs):
        super().__init__()
        # initialize with default values
        self._initialize()
        # read the given settings file
        input_dict = dict()
        settings_file = None
        if isinstance(input, str):
            ext = os.path.splitext(input)[1]
            assert ext == '.yaml', f'unrecognized file type for: {input}'
            with open(input) as fp:
                input_dict = yaml.safe_load(fp)
            #
            settings_file = input
        elif isinstance(input, dict):
            input_dict = input
        elif input is not None:
            assert False, 'got invalid input'
        #
        # override the entries with kwargs
        for k, v in kwargs.items():
            input_dict[k] = v
        #
        for key, value in input_dict.items():
            if key == 'include_files' and input_dict['include_files'] is not None:
                include_base_path = os.path.dirname(settings_file) if settings_file is not None else './'
                idict = self._parse_include_files(value, include_base_path)
                self.update(idict)
            else:
                self.__setattr__(key, value)
            #
        #
        # collect basic keys that are added during initialization
        # only these will be copied during call to basic_settings()
        self.basic_keys = list(self.keys())

    def basic_settings(self):
        '''this only returns the basic settings.
        sometimes, there is no need to copy the entire settings
        which includes the dataset_cache'''
        return ConfigDict({k:self[k] for k in self.basic_keys})
    
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    # pickling used by multiprocessing did not work without defining __getstate__
    def __getstate__(self):
        self.__dict__.copy()

    # this seems to be not required by multiprocessing
    def __setstate__(self, state):
        self.__dict__.update(state)

    def _initialize(self):
        # include additional files and merge with this dict
        self.include_files = None

    def _parse_include_files(self, include_files, include_base_path):
        input_dict = {}
        include_files = misc_utils.as_list(include_files)
        for include_file in include_files:
            append_base = not (include_file.startswith('/') and include_file.startswith('./'))
            include_file = os.path.join(include_base_path, include_file) if append_base else include_file
            with open(include_file) as ifp:
                idict = yaml.safe_load(ifp)
                input_dict.update(idict)
            #
        #
        return input_dict

# config node is derived from ConfigDict
# important node: This class handles a list specially - it will be split when the split when split() is called
# this is to support multiple decoder in a multi-task network.
# so use a tuple when specifiying array-like params for one decoder and encapsulate the tuple in a list for multi-task.
class ConfigNode(ConfigDict):
    def __init__(self):
        super().__init__()

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
