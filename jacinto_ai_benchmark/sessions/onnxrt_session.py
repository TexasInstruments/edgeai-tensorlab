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
from .. import utils
from .. import constants
from .basert_session import BaseRTSession


class ONNXRTSession(BaseRTSession):
    def __init__(self, session_name=constants.SESSION_NAME_ONNXRT, **kwargs):
        super().__init__(session_name=session_name, **kwargs)
        self.interpreter = None

    def start(self):
        super().start()

    def import_model(self, calib_data, info_dict=None):
        super().import_model(calib_data)
        # this chdir() is required for the import to work.
        interpreter_folder = os.path.join(os.environ['TIDL_BASE_PATH'], 'ti_dl/test/onnxrt')
        os.chdir(interpreter_folder)
        # create the underlying interpreter
        self.interpreter = self._create_interpreter(is_import=True)
        # check if the shape of data being provided matches with what the model expects
        if self.kwargs['input_shape'] is None:
            self.kwargs['input_shape'] = self._get_input_shape_onnxrt()
        #
        # provide the calibration data and run the import
        for c_data in calib_data:
            input_keys = list(self.kwargs['input_shape'].keys())
            c_data = utils.as_tuple(c_data)
            calib_dict = {d_name:d for d_name, d in zip(input_keys,c_data)}
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

    def _create_interpreter(self, is_import):
        # pass options to pybind
        if is_import:
            self.kwargs["delegate_options"]["import"] = "yes"
            self._cleanup_artifacts()
        else:
            self.kwargs["delegate_options"]["import"] = "no"
        #
        onnxruntime.capi._pybind_state.set_TIDLOnnxDelegate_options(self.kwargs["delegate_options"])
        sess_options = onnxruntime.SessionOptions()
        ep_list = ['TIDLExecutionProvider','CPUExecutionProvider'] #['CPUExecutionProvider']
        interpreter = onnxruntime.InferenceSession(self.kwargs['model_path'], providers=ep_list,
                                                   sess_options=sess_options)
        return interpreter

    def _set_default_options(self):
        delegate_options = self.kwargs.get("delegate_options", {})
        tidl_tools_path = os.path.join(os.environ['TIDL_BASE_PATH'], 'tidl_tools')
        required_options = {
            "tidl_tools_path": self.kwargs.get("tidl_tools_path", tidl_tools_path),
            "artifacts_folder": self.kwargs['artifacts_folder'],
            "import": self.kwargs.get("import", 'no')
        }
        optional_options = {
            "tidl_platform": "J7",
            "tidl_version": "7.2",
            "tidl_tensor_bits": self.kwargs.get("tidl_tensor_bits", 8),
            "num_tidl_subgraphs": self.kwargs.get("num_tidl_subgraphs", 16),
            "debug_level": self.kwargs.get("debug_level", 0),
            "tidl_denylist": self.kwargs.get("tidl_denylist", ""),
            "power_of_2_quantization": self.kwargs.get("power_of_2_quantization",'off'),
            "pre_batchnorm_fold": self.kwargs.get("pre_batchnorm_fold",1),
            "enable_high_resolution_optimization": self.kwargs.get("enable_high_resolution_optimization",'no'),
            "output_feature_16bit_names_list": self.kwargs.get("output_feature_16bit_names_list",''),
            "params_16bit_names_list": self.kwargs.get("params_16bit_names_list",''),
            "tidl_calibration_accuracy_level": self.kwargs.get("tidl_calibration_accuracy_level", 1),
            "reserved_compile_constraints_flag": self.kwargs.get("reserved_compile_constraints_flag",1601),
        }
        tidl_calibration_options = {k:v for k,v in self.kwargs.items() if k.startswith('tidl_calibration_options:')}
        delegate_options.update(tidl_calibration_options)
        delegate_options.update(required_options)
        delegate_options.update(optional_options)
        self.kwargs["delegate_options"] = delegate_options

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
