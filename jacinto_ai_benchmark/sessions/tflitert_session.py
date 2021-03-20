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
import numpy as np
import tflite_runtime.interpreter as tflitert_interpreter
from .. import constants
from .. import utils
from .basert_session import BaseRTSession


class TFLiteRTSession(BaseRTSession):
    def __init__(self, session_name=constants.SESSION_NAME_TFLITERT, **kwargs):
        super().__init__(session_name=session_name, **kwargs)
        self.interpreter = None

    def import_model(self, calib_data, info_dict=None):
        super().import_model(calib_data)
        
        # this chdir() is required for the import to work.
        interpreter_folder = os.path.join(os.environ['TIDL_BASE_PATH'], 'ti_dl/test/tflrt')
        os.chdir(interpreter_folder)

        self.interpreter = self._create_interpreter(is_import=True)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        for c_data in calib_data:
            c_data = utils.as_tuple(c_data)
            for c_data_entry_idx, c_data_entry in enumerate(c_data):
                self._set_tensor(input_details[c_data_entry_idx], c_data_entry)
            #
            self.interpreter.invoke()
            outputs = [self._get_tensor(output_detail) for output_detail in output_details]
        #
        return info_dict

    def start_infer(self):
        super().start_infer()
        # now create the interpreter for inference
        self.interpreter = self._create_interpreter(is_import=False)
        os.chdir(self.cwd)
        return True

    def infer_frame(self, input, info_dict=None):
        super().infer_frame(input, info_dict)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        c_data = utils.as_tuple(input)
        for c_data_entry_idx, c_data_entry in enumerate(c_data):
            self._set_tensor(input_details[c_data_entry_idx], c_data_entry)
        #
        # measure the time across only interpreter.run
        # time for setting the tensor and other overheads would be optimized out in c-api
        start_time = time.time()
        self.interpreter.invoke()
        info_dict['session_invoke_time'] = (time.time() - start_time)
        outputs = [self._get_tensor(output_detail) for output_detail in output_details]
        return outputs, info_dict

    def _create_interpreter(self, is_import):
        if is_import:
            self.kwargs["delegate_options"]["import"] = "yes"
            # make sure that the artifacts_folder is cleaneup
            self._cleanup_artifacts()
            tidl_delegate = [tflitert_interpreter.load_delegate('tidl_model_import_tflite.so', self.kwargs["delegate_options"])]
            interpreter = tflitert_interpreter.Interpreter(model_path=self.kwargs['model_path'], experimental_delegates=tidl_delegate)
        else:
            self.kwargs["delegate_options"]["import"] = "no"
            tidl_delegate = [tflitert_interpreter.load_delegate('libtidl_tfl_delegate.so', self.kwargs["delegate_options"])]
            interpreter = tflitert_interpreter.Interpreter(model_path=self.kwargs['model_path'], experimental_delegates=tidl_delegate)
        #
        interpreter.allocate_tensors()
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
            "debug_level": self.kwargs.get("debug_level", 0),
            "num_tidl_subgraphs": self.kwargs.get("num_tidl_subgraphs", 16),
            "tidl_denylist": self.kwargs.get("tidl_denylist", ""),
            "power_of_2_quantization": self.kwargs.get("power_of_2_quantization",'no'),
            "pre_batchnorm_fold": self.kwargs.get("pre_batchnorm_fold",1),
            "enable_high_resolution_optimization": self.kwargs.get("enable_high_resolution_optimization",'no'),
            "tidl_calibration_accuracy_level": self.kwargs.get("tidl_calibration_accuracy_level", 1),
            "reserved_compile_constraints_flag": self.kwargs.get("reserved_compile_constraints_flag",0),
        }
        tidl_calibration_options = {k:v for k,v in self.kwargs.items() if k.startswith('tidl_calibration_options:')}
        delegate_options.update(tidl_calibration_options)
        delegate_options.update(required_options)
        delegate_options.update(optional_options)
        self.kwargs["delegate_options"] = delegate_options

    def _get_input_shape_tflite(self):
        input_shape = {}
        model_input_details = self.interpreter.get_input_details()
        for model_input in model_input_details:
            name = model_input['name']
            shape = model_input['shape']
            input_shape.update({name:shape})
        #
        return input_shape

    def _set_tensor(self, model_input, tensor):
        if model_input['dtype'] == np.int8:
            scale, zero_point = model_input['quantization']
            tensor = np.clip(np.round(tensor/scale + zero_point), -128, 127)
            tensor = np.array(tensor, dtype=np.int8)
        elif model_input['dtype'] == np.uint8:
            scale, zero_point = model_input['quantization']
            tensor = np.clip(np.round(tensor/scale + zero_point), 0, 255)
            tensor = np.array(tensor, dtype=np.uint8)
        #
        self.interpreter.set_tensor(model_input['index'], tensor)

    def _get_tensor(self, model_output):
        tensor = self.interpreter.get_tensor(model_output['index'])
        if model_output['dtype'] == np.int8 or model_output['dtype']  == np.uint8:
            scale, zero_point = model_output['quantization']
            tensor = np.array(tensor, dtype=np.float32)
            tensor = (tensor - zero_point) / scale
        #
        return tensor
