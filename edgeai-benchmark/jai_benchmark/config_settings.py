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

import cv2
from . import utils, preprocess, postprocess, constants, sessions
from . import config_dict


class ConfigSettings(config_dict.ConfigDict):
    def __init__(self, input, **kwargs):
        super().__init__(input, **kwargs)
        # variable to pre-load datasets - so that it is not
        # separately created for each config
        self.dataset_cache = None

        # quantization params
        runtime_options = self.runtime_options if 'runtime_options' in self else dict()
        self.quantization_params = QuantizationParams(self.tensor_bits, self.calibration_frames,
                            self.calibration_iterations, is_qat=False, runtime_options=runtime_options)
        self.quantization_params_qat = QuantizationParams(self.tensor_bits, self.calibration_frames,
                            self.calibration_iterations, is_qat=True, runtime_options=runtime_options)

    def get_session_name(self, model_type_or_session_name):
        assert model_type_or_session_name in constants.MODEL_TYPES + constants.SESSION_NAMES, \
            f'get_session_cfg: input must be one of model_types: {constants.MODEL_TYPES} ' \
            f'or session_names: {constants.SESSION_NAMES}'
        if model_type_or_session_name in constants.MODEL_TYPES:
            model_type = model_type_or_session_name
            session_name = self.session_type_dict[model_type]
        else:
            session_name = model_type_or_session_name
        #
        assert session_name in constants.SESSION_NAMES, \
            f'get_session_cfg: invalid session_name: {session_name}'
        return session_name

    def get_session_type(self, model_type_or_session_name):
        session_name = self.get_session_name(model_type_or_session_name)
        return sessions.get_session_name_to_type_dict()[session_name]

    def get_runtime_options(self, model_type_or_session_name=None, is_qat=False,
                            runtime_options=None):
        # runtime_params are currently common, so session_name is currently optional
        session_name = self.get_session_name(model_type_or_session_name) if \
                model_type_or_session_name is not None else None
        quantization_params = self.quantization_params_qat if is_qat else self.quantization_params
        # runtime_options given as a keyword argument to the function overrides the default arguments
        runtime_options = quantization_params.get_runtime_options(session_name,
                                        runtime_options=runtime_options)
        return runtime_options

    def runtime_options_onnx_np2(self):
        return self.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=False,
                runtime_options={'advanced_options:quantization_scale_type': 0})

    def runtime_options_tflite_np2(self):
        return self.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=False,
                runtime_options={'advanced_options:quantization_scale_type': 0})

    def runtime_options_mxnet_np2(self):
        return self.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=False,
                runtime_options={'advanced_options:quantization_scale_type': 0})

    def runtime_options_onnx_p2(self):
        return self.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=False,
                runtime_options={'advanced_options:quantization_scale_type': 1})

    def runtime_options_tflite_p2(self):
        return self.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=False,
                runtime_options={'advanced_options:quantization_scale_type': 1})

    def runtime_options_mxnet_p2(self):
        return self.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=False,
                runtime_options={'advanced_options:quantization_scale_type': 1})

    def runtime_options_onnx_qat(self):
        return self.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=True)

    def runtime_options_tflite_qat(self):
        return self.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=True)

    def runtime_options_mxnet_qat(self):
        return self.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=True)


############################################################
# quantization / calibration params
class QuantizationParams():
    def __init__(self, tensor_bits, calibration_frames, calibration_iterations, is_qat=False, runtime_options=None):
        self.tensor_bits = tensor_bits
        self.calibration_frames = calibration_frames
        self.calibration_iterations = calibration_iterations
        self.is_qat = is_qat
        self.runtime_options = runtime_options if runtime_options is not None else dict()

    def get_calibration_iterations(self):
        # note that calibration_iterations has effect only if accuracy_level>0
        # so we can just set it to the max value here.
        # for more information see: get_calibration_accuracy_level()
        # Not overriding for 16b now
        return -1 if self.is_qat else self.calibration_iterations 

    def get_calibration_accuracy_level(self):
        # For QAT models, simple calibration is sufficient, so we shall use accuracy_level=0
        #use advance calib for 16b too
        return 0 if self.is_qat else 1 

    def get_quantization_scale_type(self):
        # 0 (non-power of 2, default), 1 (power of 2, might be helpful sometimes, needed for qat models)
        return 1 if self.is_qat else 0

    def get_runtime_options_default(self, session_name=None):
        runtime_options = {
            ##################################
            # basic_options
            #################################
            'tensor_bits': self.tensor_bits,
            'accuracy_level': self.get_calibration_accuracy_level(),
            # debug level
            'debug_level': 0,
            ##################################
            # advanced_options
            #################################
            # model optimization options
            'advanced_options:high_resolution_optimization': 0,
            'advanced_options:pre_batchnorm_fold': 1,
            # quantization/calibration options
            'advanced_options:calibration_frames': self.calibration_frames,
            # note that calibration_iterations has effect only if accuracy_level>0
            'advanced_options:calibration_iterations': self.get_calibration_iterations(),
            # 0 (non-power of 2, default), 1 (power of 2, might be helpful sometimes, needed for qat models)
            'advanced_options:quantization_scale_type': self.get_quantization_scale_type(),
            # further quantization/calibration options - these take effect
            # only if the accuracy_level in basic options is set to 9
            'advanced_options:activation_clipping': 1,
            'advanced_options:weight_clipping': 1,
            'advanced_options:bias_calibration': 1,
            'advanced_options:channel_wise_quantization': 0,
            # mixed precision options - this is just a placeholder
            # output/params names need to be specified according to a particular model
            'advanced_options:output_feature_16bit_names_list':'',
            'advanced_options:params_16bit_names_list':''
        }
        return runtime_options

    def get_runtime_options(self, session_name=None, runtime_options=None):
        # this is the default runtime_options defined above
        runtime_options_new = self.get_runtime_options_default(session_name)
        # this takes care of overrides in the code given as runtime_options keyword argument
        if runtime_options is not None:
            assert isinstance(runtime_options, dict), f'runtime_options provided via kwargs must be dict, got {type(runtime_options)}'
            runtime_options_new.update(runtime_options)
        #
        # this takes care of overrides in the settings yaml file
        runtime_options_new.update(self.runtime_options)
        return runtime_options_new
