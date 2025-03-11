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


class ConfigDict(dict):
    def __init__(self, input=None, *args, **kwargs):
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
        # override the entries with args
        for value in args:
            if isinstance(value, (dict, ConfigDict)):
                input_dict.update(value)
            #
        #
        # override the entries with kwargs
        for key, value in kwargs.items():
            input_dict[key] = value
        #
        for key, value in input_dict.items():
            if key == 'include_files' and input_dict['include_files'] is not None:
                include_base_path = os.path.dirname(settings_file) if settings_file is not None else './'
                idict = self._parse_include_files(value, include_base_path)
                self.update(idict)
            else:
                value = ConfigDict(value) if isinstance(value, dict) else value
                self.__setattr__(key, value)
            #
        #

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    # pickling used by multiprocessing did not work without defining __getstate__
    def __getstate__(self):
        self.__dict__.copy()

    # this seems to be not required by multiprocessing
    def __setstate__(self, state):
        self.__dict__.update(state)

    def _initialize(self):
        pass

    def _parse_include_files(self, include_files, include_base_path):
        input_dict = {}
        include_files = utils.as_list(include_files)
        for include_file in include_files:
            append_base = not (include_file.startswith('/') and include_file.startswith('./'))
            include_file = os.path.join(include_base_path, include_file) if append_base else include_file
            with open(include_file) as ifp:
                idict = yaml.safe_load(ifp)
                input_dict.update(idict)
            #
        #
        return input_dict

    def update(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, (dict, ConfigDict)):
                self.update(**arg)
            #
        #
        for k, v in kwargs.items():
            if k in self:
                if isinstance(v, (dict, ConfigDict)) and k in self and isinstance(self[k], (dict, ConfigDict)):
                    v = ConfigDict(v) if isinstance(v, dict) else v
                    self[k].update(v)
                else:
                    self[k] = v
                #
            else:
                v = ConfigDict(v) if isinstance(v, dict) else v
                self[k] = v
            #
        #
        return self
