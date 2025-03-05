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

from ..core import TVMDLRRuntimeWrapper

import os
import time
import copy

from .. import constants
from ..import utils
from .basert_session import BaseRTSession


class TVMDLRSession(BaseRTSession, TVMDLRRuntimeWrapper):
    def __init__(self, session_name=constants.SESSION_NAME_TVMDLR, **kwargs):
        BaseRTSession.__init__(self, session_name=session_name, **kwargs)
        TVMDLRRuntimeWrapper.__init__(self)
        self.kwargs['input_data_layout'] = self.kwargs.get('input_data_layout', constants.NCHW)
        self.supported_machines = (constants.TARGET_MACHINE_PC_EMULATION, constants.TARGET_MACHINE_EVM)
        target_machine = self.kwargs['target_machine']
        assert target_machine in self.supported_machines, f'invalid target_machine {target_machine}'

    def start(self):
        BaseRTSession.start(self)
        TVMDLRRuntimeWrapper.start(self)

    def import_model(self, calib_data, info_dict=None):
        # prepare for actual model import
        BaseRTSession._prepare_for_import(self)

        calib_list = []
        for input_data in calib_data:
            # input_data = self._format_input_data(input_data)
            if not isinstance(input_data, tuple):
                input_data = (input_data,)
            #
            if self.input_normalizer is not None:
                input_data, _ = self.input_normalizer(input_data, {})
            #
            calib_list.append(input_data)
        #

        TVMDLRRuntimeWrapper._prepare_for_import(self, calib_list)
        os.chdir(self.cwd)
        return info_dict

    def start_infer(self):
        BaseRTSession._prepare_for_inference(self)
        TVMDLRRuntimeWrapper._prepare_for_inference(self)
        os.chdir(self.cwd)
        return True

    def infer_frame(self, input_data, info_dict=None):
        super().infer_frame(input_data, info_dict)

        # input_data = self._format_input_data(input_data)
        if not isinstance(input_data, tuple):
            input_data = (input_data,)
        #

        if self.input_normalizer is not None:
            input_data, _ = self.input_normalizer(input_data, {})
        #

        start_time = time.time()
        outputs = TVMDLRRuntimeWrapper._run(self, input_data)
        info_dict['session_invoke_time'] = (time.time() - start_time)
        self._update_output_details(outputs)
        return outputs, info_dict


if __name__ == '__main__':
    tvm_model = TVMDLRSession()
