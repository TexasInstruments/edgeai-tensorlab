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
import numpy as np
import warnings
import onnxruntime

try:
    import onnx
except:
    #warnings.warn('onnx could not be imported - this is not required for inference, but may be required for import')
    pass

from .. import utils
from .. import constants
from .basert_session import BaseRTSession


class ONNXRTSession(BaseRTSession):
    def __init__(self, session_name=constants.SESSION_NAME_ONNXRT, **kwargs):
        super().__init__(session_name=session_name, **kwargs)
        self.kwargs['input_data_layout'] = self.kwargs.get('input_data_layout', constants.NCHW)
        self.interpreter = None

    def start(self):
        super().start()

    def import_model(self, calib_data, info_dict=None):
        super().import_model(calib_data)

        # create the underlying interpreter
        self.interpreter = self._create_interpreter(is_import=True)
        # check if the shape of data being provided matches with what the model expects
        if self.kwargs['input_shape'] is None:
            self.kwargs['input_shape'] = self._get_input_shape_onnxrt()
        #
        # provide the calibration data and run the import
        for in_data in calib_data:
            input_keys = list(self.kwargs['input_shape'].keys())
            in_data = utils.as_tuple(in_data)
            if self.input_normalizer is not None:
                in_data, _ = self.input_normalizer(in_data, {})
            #
            calib_dict = {d_name:d for d_name, d in zip(input_keys,in_data)}
            # model may need additional inputs given in extra_inputs
            if self.kwargs['extra_inputs'] is not None:
                calib_dict.update(self.kwargs['extra_inputs'])
            #
            output_keys = list(self.kwargs['output_shape'].keys()) \
                if self.kwargs['output_shape'] is not None else None
            # run the actual import step
            outputs = self.interpreter.run(output_keys, calib_dict)
        #
        return info_dict

    def start_infer(self):
        super().start_infer()
        # create the underlying interpreter
        self.interpreter = self._create_interpreter(is_import=False)
        # input_shape is needed during inference - get it if it is not given
        if self.kwargs['input_shape'] is None:
            self.kwargs['input_shape'] = self._get_input_shape_onnxrt()
        #
        os.chdir(self.cwd)
        return True

    def infer_frame(self, input, info_dict=None):
        super().infer_frame(input, info_dict)
        input_keys = list(self.kwargs['input_shape'].keys())
        in_data = utils.as_tuple(input)
        if self.input_normalizer is not None:
            in_data, _ = self.input_normalizer(in_data, {})
        #
        input_dict = {d_name:d for d_name, d in zip(input_keys,in_data)}
        # model needs additional inputs given in extra_inputs
        if self.kwargs['extra_inputs'] is not None:
            input_dict.update(self.kwargs['extra_inputs'])
        #
        # output_shape is not mandatory, output_keys can be None
        output_keys = list(self.kwargs['output_shape'].keys()) \
            if self.kwargs['output_shape'] is not None else None
        # run the actual inference
        start_time = time.time()
        outputs = self.interpreter.run(output_keys, input_dict)
        info_dict['session_invoke_time'] = (time.time() - start_time)
        return outputs, info_dict

    def set_runtime_option(self, option, value):
        self.kwargs["runtime_options"][option] = value

    def get_runtime_option(self, option, default=None):
        return self.kwargs["runtime_options"].get(option, default)

    def _create_interpreter(self, is_import):
        # pass options to pybind
        if is_import:
            self.kwargs["runtime_options"]["import"] = "yes"
        else:
            self.kwargs["runtime_options"]["import"] = "no"
        #
        runtime_options = self.kwargs["runtime_options"]
        sess_options = onnxruntime.SessionOptions()

        if self.kwargs['tidl_offload']:
            ep_list = ['TIDLCompilationProvider', 'CPUExecutionProvider'] if is_import else \
                      ['TIDLExecutionProvider', 'CPUExecutionProvider']
            interpreter = onnxruntime.InferenceSession(self.kwargs['model_file'], providers=ep_list,
                            provider_options=[runtime_options, {}], sess_options=sess_options)
        else:
            ep_list = ['CPUExecutionProvider']
            interpreter = onnxruntime.InferenceSession(self.kwargs['model_file'], providers=ep_list,
                            provider_options=[{}], sess_options=sess_options)
        #
        return interpreter

    def _set_default_options(self):
        runtime_options = self.kwargs.get("runtime_options", {})
        default_options = {
            "platform": constants.TIDL_PLATFORM,
            "version": constants.TIDL_VERSION_STR,
            "tidl_tools_path": self.kwargs["tidl_tools_path"],
            "artifacts_folder": self.kwargs["artifacts_folder"],
            "tensor_bits": self.kwargs.get("tensor_bits", 8),
            "import": self.kwargs.get("import", 'no'),
            # note: to add advanced options here, start it with 'advanced_options:'
            # example 'advanced_options:pre_batchnorm_fold':1
        }
        default_options.update(runtime_options)
        self.kwargs["runtime_options"] = default_options

    def _get_input_shape_onnxrt(self):
        input_details = self.interpreter.get_inputs()
        input_shape = {}
        for inp in input_details:
            input_shape.update({inp.name:inp.shape})
        #
        return input_shape

    def _get_output_shape_onnxrt(self):
        output_details = self.interpreter.get_outputs()
        output_shape = {}
        for oup in output_details:
            output_shape.update({oup.name:oup.shape})
        #
        return output_shape
