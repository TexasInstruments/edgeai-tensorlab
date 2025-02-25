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

import enum
import copy
import sys
import cv2
import numpy as np

from . import utils, preprocess, postprocess, constants, sessions
from . import config_dict


class RuntimeOptions(config_dict.ConfigDict):
    def __init__(self, input, **kwargs):
        super().__init__(input, **kwargs)
        # target device presets
        preset_dict = None
        if isinstance(self.target_device_preset, dict):
            preset_dict = self.target_device_preset
        elif self.target_device_preset and self.target_device:
            preset_dict = constants.TARGET_DEVICE_SETTINGS_PRESETS[self.target_device]
        #
        if preset_dict:
            self.update(preset_dict)
        #

    def _get_runtime_options_default(self, session_name=None, is_qat=False,
            min_options=None, max_options=None, fast_calibration=True, **kwargs):
        '''
        Default runtime options.
        Overiride this according to the needs of specific configs using methods below.

        Args:
            session_name: onnxrt, tflitert or tvmdlr
            qat_type: set appropriately for QAT models
            det_options: True for detection models, False for other models. Can also be a dictionary.

        Returns: runtime_options
        '''
        advanced_options_quantization_scale_type = kwargs.get('advanced_options:quantization_scale_type', None)
        advanced_options_prequantized_model = kwargs.get('advanced_options:prequantized_model', constants.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_NONE)

        calibration_iterations_factor = self._get_calibration_iterations_factor(fast_calibration)

        min_options = min_options or dict()
        max_options = max_options or dict()

        calibration_frames = max(int(self.calibration_frames * calibration_iterations_factor), 1)
        calibration_frames = np.clip(calibration_frames, min_options.get('advanced_options:calibration_frames', -sys.maxsize), max_options.get('advanced_options:calibration_frames', sys.maxsize))

        calibration_iterations = max(int(self._get_calibration_iterations(advanced_options_quantization_scale_type, is_qat, advanced_options_prequantized_model) * calibration_iterations_factor), 1)
        calibration_iterations = np.clip(calibration_iterations, min_options.get('advanced_options:calibration_iterations', -sys.maxsize), max_options.get('advanced_options:calibration_iterations', sys.maxsize))

        runtime_options = {
            ##################################
            # basic_options
            #################################
            'tensor_bits': self.tensor_bits,
            'accuracy_level': self._get_calibration_accuracy_level(advanced_options_quantization_scale_type, is_qat, advanced_options_prequantized_model),
            # debug level
            'debug_level': 0,
            'inference_mode': 0,
            ##################################
            # advanced_options
            #################################
            # model optimization options
            'advanced_options:high_resolution_optimization': 0,
            'advanced_options:pre_batchnorm_fold': 1,
            # quantization/calibration options
            'advanced_options:calibration_frames': calibration_frames,
            # note that calibration_iterations has effect only if accuracy_level>0
            'advanced_options:calibration_iterations': calibration_iterations,
            # 0 (non-power of 2, default), 1 (power of 2, might be helpful sometimes, needed for qat models)
            'advanced_options:quantization_scale_type': self._get_quantization_scale_type(advanced_options_quantization_scale_type, is_qat, advanced_options_prequantized_model),
            # further quantization/calibration options - these take effect
            # only if the accuracy_level in basic options is set to 9
            'advanced_options:activation_clipping': 1,
            'advanced_options:weight_clipping': 1,
            # if bias_clipping is set to 0 (default), weight scale will be adjusted to avoid bias_clipping
            # if bias_clipping is set to 1, weight scale is computed solely based on weight range.
            # this should only affect the mode where the bias is clipped to 16bits (default in TDA4VM).
            #'advanced_options:bias_clipping': 1,
            'advanced_options:bias_calibration': 1,
            # when quantization_scale_type is 4, what is set here is IGNORED and per channel quantization is ALWAYS used
            # for other values of quantization_scale_type, we can set this to elable per channel quantization
            #'advanced_options:channel_wise_quantization': 0,
            # mixed precision options - this is just a placeholder
            # output/params names need to be specified according to a particular model
            'advanced_options:output_feature_16bit_names_list':'',
            'advanced_options:params_16bit_names_list':'',
            # optimize data conversion options by moving them from arm to c7x
            'advanced_options:add_data_convert_ops': 3,
            # max number of nodes in a subgraph (default is 750)
            #'advanced_options:max_num_subgraph_nodes': 2000,
            # use a specific firmware version
            'advanced_options:c7x_firmware_version': constants.TIDL_FIRMWARE_VERSION,
            ##################################
            # additional options (internal / performance estimation)
            #################################
            "ti_internal_nc_flag" : 83886080, #1601
        }

        # kwargs will have this already
        # advanced_options_prequantized_model_clip = (is_qat and advanced_options_prequantized_model == constants.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_CLIP)
        # advanced_options_prequantized_model_qdq = (is_qat and advanced_options_prequantized_model == constants.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_QDQ)
        # if advanced_options_prequantized_model_qdq:
        #     runtime_options.update({'advanced_options:prequantized_model': 1})
        # #

        # update with kwargs
        runtime_options.update(kwargs)

        return runtime_options

    def get_runtime_options(self, model_type_or_session_name=None, advanced_options_quantization_scale_type=None, is_qat=False,
            det_options=None, ext_options=None, min_options=None, max_options=None, **kwargs):
        '''
        example usage for min_options and max_options to set the limit
            settings.runtime_options_onnx_np2(max_options={'advanced_options:calibration_frames':25, 'advanced_options:calibration_iterations':25})
             similarly min_options can be used to set lower limit
             currently only calibration_frames and calibration_iterations are handled in this function.
        '''
        # runtime_params are currently common, so session_name is currently optional
        session_name = self.get_session_name(model_type_or_session_name) if \
                model_type_or_session_name is not None else None

        # this is the default runtime_options defined above
        runtime_options = self._get_runtime_options_default(session_name=session_name, is_qat=is_qat,
            min_options=min_options, max_options=max_options, **kwargs)

        # this takes care of overrides given as ext_options keyword argument
        if ext_options is not None:
            assert isinstance(ext_options, dict), \
                f'runtime_options provided via kwargs must be dict, got {type(ext_options)}'
            runtime_options.update(ext_options)
        #

        object_detection_meta_arch_type = runtime_options.get('object_detection:meta_arch_type', None)

        # for tflite models, these options are directly handled inside tidl
        # for onnx od models, od post proc options are specified in the prototxt and it is modified with these options
        # use a large top_k, keep_top_k and low confidence_threshold for accuracy measurement
        # SSD models have a high detection_threshold as default since thier runtime is sensitive to this threhold
        if det_options == 'SSD' or (det_options is True and object_detection_meta_arch_type in constants.TIDL_DETECTION_META_ARCH_TYPE_SSD_LIST):
            if self.detection_threshold is True:
                runtime_options.update({
                    'object_detection:confidence_threshold': 0.3,
                })
            #
            if self.detection_top_k is True:
                runtime_options.update({
                    'object_detection:top_k': 200,
                })
            #
            if self.detection_nms_threshold is True:
                runtime_options.update({
                    'object_detection:nms_threshold': 0.45,
                })
            #
            if self.detection_keep_top_k is True:
                runtime_options.update({
                    'object_detection:keep_top_k': 200
                })
            #
        elif det_options is True:
            # other models, especially YOLO modles can use a lower detection threshold
            if self.detection_threshold is True:
                runtime_options.update({
                    'object_detection:confidence_threshold': 0.05,
                })
            #
            if self.detection_top_k is True:
                runtime_options.update({
                    'object_detection:top_k': 500,
                })
            #
            if self.detection_nms_threshold is True:
                runtime_options.update({
                    'object_detection:nms_threshold': 0.45,
                })
            #
            if self.detection_keep_top_k is True:
                runtime_options.update({
                    'object_detection:keep_top_k': 200
                })
            #
        elif isinstance(det_options, dict):
            runtime_options.update(det_options)
        #

        # this is now taken care of here, instead of in the below functions
        if self.runtime_options is not None:
            runtime_options.update(self.runtime_options)
        #
        return runtime_options

    def runtime_options_onnx_np2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = kwargs.get('advanced_options:quantization_scale_type', constants.QUANTScaleType.QUANT_SCALE_TYPE_NP2)
        return self.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=False, **kwargs)

    def runtime_options_tflite_np2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = kwargs.get('advanced_options:quantization_scale_type', constants.QUANTScaleType.QUANT_SCALE_TYPE_NP2)
        return self.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=False, **kwargs)

    def runtime_options_mxnet_np2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = kwargs.get('advanced_options:quantization_scale_type', constants.QUANTScaleType.QUANT_SCALE_TYPE_NP2)
        return self.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=False, **kwargs)

    def runtime_options_onnx_p2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = kwargs.get('advanced_options:quantization_scale_type', constants.QUANTScaleType.QUANT_SCALE_TYPE_P2)
        return self.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=False, **kwargs)

    def runtime_options_tflite_p2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = kwargs.get('advanced_options:quantization_scale_type', constants.QUANTScaleType.QUANT_SCALE_TYPE_P2)
        return self.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=False, **kwargs)

    def runtime_options_mxnet_p2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = kwargs.get('advanced_options:quantization_scale_type', constants.QUANTScaleType.QUANT_SCALE_TYPE_P2)
        return self.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=False, **kwargs)

    def runtime_options_onnx_qat_v1(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = kwargs.get('advanced_options:quantization_scale_type', constants.QUANTScaleType.QUANT_SCALE_TYPE_P2)
        kwargs['advanced_options:prequantized_model'] = kwargs.get('advanced_options:prequantized_model', constants.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_CLIP)
        return self.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=True, **kwargs)

    def runtime_options_tflite_qat_v1(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = kwargs.get('advanced_options:quantization_scale_type', constants.QUANTScaleType.QUANT_SCALE_TYPE_P2)
        kwargs['advanced_options:prequantized_model'] = kwargs.get('advanced_options:prequantized_model', constants.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_CLIP)
        return self.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=True, **kwargs)

    def runtime_options_mxnet_qat_v1(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = kwargs.get('advanced_options:quantization_scale_type', constants.QUANTScaleType.QUANT_SCALE_TYPE_P2)
        kwargs['advanced_options:prequantized_model'] = kwargs.get('advanced_options:prequantized_model', constants.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_CLIP)
        return self.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=True, **kwargs)

    def runtime_options_onnx_qat_v2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = kwargs.get('advanced_options:quantization_scale_type', constants.QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN)
        kwargs['advanced_options:prequantized_model'] = kwargs.get('advanced_options:prequantized_model', constants.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_QDQ)
        return self.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=True, **kwargs)

    def runtime_options_onnx_qat_v2_p2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = kwargs.get('advanced_options:quantization_scale_type', constants.QUANTScaleType.QUANT_SCALE_TYPE_P2)
        kwargs['advanced_options:prequantized_model'] = kwargs.get('advanced_options:prequantized_model', constants.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_QDQ)
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

    def _get_calibration_iterations(self, advanced_options_quantization_scale_type, is_qat, advanced_options_prequantized_model):
        # note that calibration_iterations has effect only if accuracy_level>0
        # so we can just set it to the max value here.
        # for more information see: get_calibration_accuracy_level()
        # Not overriding for 16b now
        quantized_model = is_qat or advanced_options_prequantized_model != constants.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_NONE
        return -1 if quantized_model else self.calibration_iterations

    def _get_calibration_accuracy_level(self, advanced_options_quantization_scale_type, is_qat, advanced_options_prequantized_model):
        # For QAT models, simple calibration is sufficient, so we shall use accuracy_level=0
        #use advance calib for 16b too
        quantized_model = is_qat or advanced_options_prequantized_model != constants.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_NONE
        return 0 if quantized_model else 1

    def _get_quantization_scale_type(self, advanced_options_quantization_scale_type, is_qat, advanced_options_prequantized_model):
        # 0 (non-power of 2, default)
        # 1 (power of 2, might be helpful sometimes, needed for p2 qat models)
        # 3 (non-power of 2 qat/prequantized model, supported in newer devices)
        # 4 (non-power2 of 2, supported in newer devices)
        return advanced_options_quantization_scale_type.value if isinstance(advanced_options_quantization_scale_type, enum.Enum) else advanced_options_quantization_scale_type

    def _get_calibration_iterations_factor(self, fast_calibration):
        # model may need higher number of itarations for certain devices (i.e. when per channel quantization is not supported)
        device_needs_more_iterations = (self.calibration_iterations_factor is not None)
        model_needs_more_iterations = (not fast_calibration)
        if device_needs_more_iterations and model_needs_more_iterations:
            return self.calibration_iterations_factor
        else:
            return constants.CALIBRATION_ITERATIONS_FACTOR_1X


