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

from .. import utils
from .. import constants
from .basert_session import BaseRTSession


class ONNXRTSession(BaseRTSession):
    def __init__(self, session_name=constants.SESSION_NAME_ONNXRT, **kwargs):
        super().__init__(session_name=session_name, **kwargs)
        self.kwargs['input_data_layout'] = self.kwargs.get('input_data_layout', constants.NCHW)

    def start(self):
        super().start()

    def import_model(self, calib_data, info_dict=None):
        super().import_model(calib_data)

        # create the underlying interpreter
        self.interpreter = self._create_interpreter(is_import=True)

        self._get_input_output_details_onnx(self.interpreter)

        # provide the calibration data and run the import
        for frame_idx, in_data in enumerate(calib_data):
            calib_dict = self.get_in_dict(in_data)    

            if self.input_normalizer is not None:
                calib_dict, _ = self.input_normalizer(calib_dict, {})
            
            # model may need additional inputs given in extra_inputs
            if self.kwargs['extra_inputs'] is not None:
                calib_dict.update(self.kwargs['extra_inputs'])
            #
            output_keys = [getattr(d_info, 'name') for d_info in self.interpreter.get_outputs()] \
                if self.kwargs['output_details'] is not None else None
            # run the actual import step
            outputs = self.interpreter.run(output_keys, calib_dict)
            self._update_output_details(outputs)
        #

        print("================================ import model =============")
        return info_dict

    def start_infer(self):
        super().start_infer()
        # create the underlying interpreter
        self.interpreter = self._create_interpreter(is_import=False)
        # input_details is needed during inference - get it if it is not given
        self._get_input_output_details_onnx(self.interpreter)
        os.chdir(self.cwd)
        return True

    def get_in_dict(self, in_data):
        if not isinstance(in_data, list) and not isinstance(in_data, dict):
            in_data = utils.as_tuple(in_data)        

        if isinstance(in_data, dict):
            return in_data
        
        return {getattr(d_info, 'name'):d for d_info, d in zip(self.interpreter.get_inputs(),in_data)}
        

    def infer_frame(self, input, info_dict=None):
        super().infer_frame(input, info_dict)

        input_dict = self.get_in_dict(input)

        if self.input_normalizer is not None:
            input_dict, _ = self.input_normalizer(input_dict, {})

        # model needs additional inputs given in extra_inputs
        if self.kwargs['extra_inputs'] is not None:
            input_dict.update(self.kwargs['extra_inputs'])
        #
        # output_details is not mandatory, output_keys can be None
        output_keys = [getattr(d_info, 'name') for d_info in self.interpreter.get_outputs()] \
            if self.kwargs['output_details'] is not None else None
        # run the actual inference
        start_time = time.time()
        outputs = self.interpreter.run(output_keys, input_dict)
        info_dict['session_invoke_time'] = (time.time() - start_time)
        self._update_output_details(outputs)
        return outputs, info_dict

    def set_runtime_option(self, option, value):
        self.kwargs["runtime_options"][option] = value

    def get_runtime_option(self, option, default=None):
        return self.kwargs["runtime_options"].get(option, default)

    def _create_interpreter(self, is_import):
        # move the import inside the function, so that onnxruntime needs to be installed
        # only if some one wants to use it
        import onnxruntime
        # pass options to pybind
        if is_import:
            self.kwargs["runtime_options"]["import"] = "yes"
        else:
            self.kwargs["runtime_options"]["import"] = "no"
        #
        runtime_options = self.kwargs["runtime_options"]
        sess_options = onnxruntime.SessionOptions()
        
        onnxruntime_graph_optimization_level = self.kwargs["runtime_options"].get('onnxruntime:graph_optimization_level', None)
        if onnxruntime_graph_optimization_level is not None:
            # for transformer models, it is necessary to set graph_optimization_level in session options for onnxruntime
            # to onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL so that TIDL can properly handle the model.
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        # suppress warnings
        sess_options.log_severity_level = 3

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
