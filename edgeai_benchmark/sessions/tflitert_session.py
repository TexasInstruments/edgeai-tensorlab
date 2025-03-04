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
import time
import warnings
import copy
import struct
import numpy as np

from ..runtimes import TFLiteRuntimeWrapper
from .. import constants
from .. import utils
from .basert_session import BaseRTSession


class TFLiteRTSession(BaseRTSession, TFLiteRuntimeWrapper):
    def __init__(self, session_name=constants.SESSION_NAME_TFLITERT, **kwargs):
        BaseRTSession.__init__(self, session_name=session_name, **kwargs)
        TFLiteRuntimeWrapper.__init__(self)
        self.kwargs['input_data_layout'] = self.kwargs.get('input_data_layout', constants.NHWC)

    def start(self):
        super().start()
        TFLiteRuntimeWrapper.start(self)

    def import_model(self, calib_data, info_dict=None):
        super().import_model(calib_data)

        # create the underlying interpreter
        self.prepare_for_import()

        for frame_id, in_data in enumerate(calib_data):
            in_data = self._format_input_data(in_data)
            if self.input_normalizer is not None:
                in_data, _ = self.input_normalizer(in_data, {})
            #
            outputs = self.run(in_data)
            self._update_output_details(outputs)
        #
        return info_dict

    def start_infer(self):
        super().start_infer()
        # now create the interpreter for inference
        self.prepare_for_inference()
        os.chdir(self.cwd)
        return True

    def infer_frame(self, input, info_dict=None):
        super().infer_frame(input, info_dict)

        in_data = self._format_input_data(input)

        if self.input_normalizer is not None:
            in_data, _ = self.input_normalizer(in_data, {})
        #

        # measure the time across only interpreter.run
        # time for setting the tensor and other overheads would be optimized out in c-api
        start_time = time.time()
        outputs = self.run(in_data)
        info_dict['session_invoke_time'] = (time.time() - start_time)
        self._update_output_details(outputs)
        return outputs, info_dict
