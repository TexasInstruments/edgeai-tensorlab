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


from . import runners
from . import constants, datasets, sessions


class ConfigSettings(runners.GetRuntimeOptions):
    def __init__(self, input, **kwargs):
        super().__init__(input, **kwargs)
        # variable to pre-load datasets - so that it is not separately created for each config
        self.dataset_cache = datasets.initialize_datasets(self)

    def runtime_options_onnx_np2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = self.runtime_options.get('advanced_options:quantization_scale_type', constants.QUANTScaleType.QUANT_SCALE_TYPE_NP2)
        return self.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=False, **kwargs)

    def runtime_options_tflite_np2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = self.runtime_options.get('advanced_options:quantization_scale_type', constants.QUANTScaleType.QUANT_SCALE_TYPE_NP2)
        return self.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=False, **kwargs)

    def runtime_options_mxnet_np2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = self.runtime_options.get('advanced_options:quantization_scale_type', constants.QUANTScaleType.QUANT_SCALE_TYPE_NP2)
        return self.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=False, **kwargs)

    def runtime_options_onnx_p2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = self.runtime_options.get('advanced_options:quantization_scale_type', constants.QUANTScaleType.QUANT_SCALE_TYPE_P2)
        return self.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=False, **kwargs)

    def runtime_options_tflite_p2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = self.runtime_options.get('advanced_options:quantization_scale_type', constants.QUANTScaleType.QUANT_SCALE_TYPE_P2)
        return self.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=False, **kwargs)

    def runtime_options_mxnet_p2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = self.runtime_options.get('advanced_options:quantization_scale_type', constants.QUANTScaleType.QUANT_SCALE_TYPE_P2)
        return self.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=False, **kwargs)

    def runtime_options_onnx_qat_v1(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = constants.QUANTScaleType.QUANT_SCALE_TYPE_P2
        # kwargs['advanced_options:prequantized_model'] = constants.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_CLIP
        return self.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=True, **kwargs)

    def runtime_options_tflite_qat_v1(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = constants.QUANTScaleType.QUANT_SCALE_TYPE_P2
        # kwargs['advanced_options:prequantized_model'] = constants.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_CLIP
        return self.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=True, **kwargs)

    def runtime_options_mxnet_qat_v1(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = constants.QUANTScaleType.QUANT_SCALE_TYPE_P2
        # kwargs['advanced_options:prequantized_model'] = constants.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_CLIP
        return self.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=True, **kwargs)

    def runtime_options_onnx_qat_v2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = constants.QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN
        kwargs['advanced_options:prequantized_model'] = constants.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_QDQ
        return self.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=True, **kwargs)

    def runtime_options_onnx_qat_v2_p2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = constants.QUANTScaleType.QUANT_SCALE_TYPE_P2
        kwargs['advanced_options:prequantized_model'] = constants.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_QDQ
        return self.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=True, **kwargs)

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


class CustomConfigSettings(ConfigSettings):
    def __init__(self, input, dataset_loading=False, **kwargs):
        super().__init__(input, dataset_loading=dataset_loading, **kwargs)
