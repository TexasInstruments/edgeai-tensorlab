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
import os
import sys
import warnings

import cv2
import numpy as np
import yaml

from . import config_dict
from . import presets


class ConfigRuntimeOptions(config_dict.ConfigDict):
    def __init__(self, input, **kwargs):
        super().__init__(input, **kwargs)
        if not hasattr(self, 'runtime_options'):
            self.runtime_options = {}
        #
        if not hasattr(self, 'calibration_iterations_factor'):
            self.calibration_iterations_factor = 1.0
        #
        if not hasattr(self, 'c7x_firmware_version'):
            self.c7x_firmware_version = None
        #

        # target device presets
        preset_dict = None
        if isinstance(self.target_device_preset, dict):
            preset_dict = self.target_device_preset
        elif self.target_device_preset and self.target_device:
            preset_dict = presets.TARGET_DEVICE_SETTINGS_PRESETS[self.target_device]
        #
        if preset_dict:
            self.update(preset_dict)
        #

    def _get_runtime_options_default(self, model_type_or_session_name=None, is_qat=False,
            min_options=None, max_options=None, fast_calibration=True, **kwargs):
        '''
        Default runtime options.
        Overiride this according to the needs of specific configs using methods below.

        Args:
            model_type_or_session_name: onnxrt, tflitert or tvmdlr
            is_qat: set appropriately for QAT models

        Returns: runtime_options
        '''
        # pop advanced_options:quantization_scale_type out and set it - just to be clear
        advanced_options_quantization_scale_type = kwargs.pop('advanced_options:quantization_scale_type',
            self.runtime_options.get('advanced_options:quantization_scale_type', presets.QUANTScaleType.QUANT_SCALE_TYPE_P2)
        )

        calibration_iterations_factor = self._get_calibration_iterations_factor(fast_calibration)

        min_options = min_options or dict()
        max_options = max_options or dict()

        calibration_frames = max(int(self.calibration_frames * calibration_iterations_factor), 1)
        calibration_frames = np.clip(calibration_frames, min_options.get('advanced_options:calibration_frames', -sys.maxsize), max_options.get('advanced_options:calibration_frames', sys.maxsize))

        calibration_iterations = max(int(self._get_calibration_iterations(is_qat) * calibration_iterations_factor), 1)
        calibration_iterations = np.clip(calibration_iterations, min_options.get('advanced_options:calibration_iterations', -sys.maxsize), max_options.get('advanced_options:calibration_iterations', sys.maxsize))

        runtime_options = {
            ##################################
            # basic_options
            #################################
            'tensor_bits': self.tensor_bits,
            'accuracy_level': self._get_calibration_accuracy_level(is_qat),
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
            # 4 (per-channel quantization)
            'advanced_options:quantization_scale_type': advanced_options_quantization_scale_type,
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
            ##################################
            # additional options (internal / performance estimation)
            "ti_internal_nc_flag" : 83886080, #1601
        }

        # additional options (firmware version)
        if self.c7x_firmware_version is not None and self.c7x_firmware_version != "":
            runtime_options.update({
                'advanced_options:c7x_firmware_version': self.c7x_firmware_version
            })

            warnings.warn(f'\nINFO: advanced_options:c7x_firmware_version passed to tidl_tools from this repo for model compilation is: {self.c7x_firmware_version}'
                          f'\nINFO: for potential firmware update needed in SDK to run this model, see the SDK version compatibiltiy table: '
                          f'\nINFO: https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/version_compatibility_table.md')
        #

        # set other runtime_options from kwargs
        runtime_options.update(kwargs)
        return runtime_options

    def get_runtime_options(self, model_type_or_session_name=None, is_qat=False,
            det_options=None, ext_options=None, min_options=None, max_options=None, **kwargs):
        '''
        example usage for min_options and max_options to set the limit
            settings.runtime_options_onnx_np2(max_options={'advanced_options:calibration_frames':25, 'advanced_options:calibration_iterations':25})
             similarly min_options can be used to set lower limit
             currently only calibration_frames and calibration_iterations are handled in this function.

        model_type_or_session_name: runtime_params are currently common, so session_name is currently not needed here
        det_options: True for detection models, False for other models. Can also be a dictionary.
        '''

        runtime_options = copy.deepcopy(self.runtime_options or {})

        # this is the default runtime_options defined above
        runtime_options_override = self._get_runtime_options_default(
            model_type_or_session_name=model_type_or_session_name, is_qat=is_qat,
            min_options=min_options, max_options=max_options, **kwargs)

        runtime_options.update(runtime_options_override)

        # this takes care of overrides given as ext_options keyword argument
        if ext_options is not None:
            assert isinstance(ext_options, dict), \
                f'runtime_options provided via kwargs must be dict, got {type(ext_options)}'
            runtime_options.update(ext_options)
        #

        object_detection_meta_arch_type = runtime_options.get('object_detection:meta_arch_type', None)

        # for tflite models, these options are directly processed inside tidl
        # for onnx od models, od post proc options are specified in the prototxt and it is modified with these options
        # use a large top_k, keep_top_k and low confidence_threshold for accuracy measurement
        if isinstance(det_options, dict):
            runtime_options.update(det_options)
        elif det_options:
            # SSD models may need to have a high detection_threshold afor inference since thier runtime is sensitive to this threhold
            is_ssd = det_options == 'SSD' or (det_options is True and object_detection_meta_arch_type in presets.TIDL_DETECTION_META_ARCH_TYPE_SSD_LIST)
            detection_threshold_default = (0.3 if is_ssd else 0.05)
            detection_top_k_default = (200 if is_ssd else 500)
            if self.detection_threshold:
                runtime_options.update({
                    'object_detection:confidence_threshold': (detection_threshold_default if self.detection_threshold is True else self.detection_threshold),
                })
            #
            if self.detection_top_k:
                runtime_options.update({
                    'object_detection:top_k': (detection_top_k_default if self.detection_top_k is True else self.detection_top_k),
                })
            #
            if self.detection_nms_threshold:
                runtime_options.update({
                    'object_detection:nms_threshold': (0.45 if self.detection_nms_threshold is True else self.detection_nms_threshold),
                })
            #
            if self.detection_keep_top_k:
                runtime_options.update({
                    'object_detection:keep_top_k': (200 if self.detection_keep_top_k is True else self.detection_keep_top_k)
                })
            #
        #
        return runtime_options

    def _get_calibration_iterations(self, is_qat):
        # note that calibration_iterations has effect only if accuracy_level>0
        # so we can just set it to the max value here.
        # for more information see: get_calibration_accuracy_level()
        # Not overriding for 16b now
        return -1 if is_qat else self.calibration_iterations

    def _get_calibration_accuracy_level(self, is_qat):
        # For QAT models, simple calibration is sufficient, so we shall use accuracy_level=0
        #use advance calib for 16b too
        return 0 if is_qat else 1

    def _get_calibration_iterations_factor(self, fast_calibration):
        # model may need higher number of itarations for certain devices (i.e. when per channel quantization is not supported)
        device_needs_more_iterations = (self.calibration_iterations_factor is not None)
        model_needs_more_iterations = (not fast_calibration)
        if device_needs_more_iterations and model_needs_more_iterations:
            return self.calibration_iterations_factor
        else:
            return presets.CALIBRATION_ITERATIONS_FACTOR_1X

    def runtime_options_onnx_np2(self, **kwargs):
        advanced_options_quantization_scale_type_default = self.runtime_options.get('advanced_options:quantization_scale_type', None)
        # do not modify the scale type if per channel quantization is supported in the device
        if advanced_options_quantization_scale_type_default != presets.QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN:
            kwargs['advanced_options:quantization_scale_type'] = presets.QUANTScaleType.QUANT_SCALE_TYPE_NP2
        #
        return self.get_runtime_options(presets.MODEL_TYPE_ONNX, is_qat=False, **kwargs)

    def runtime_options_tflite_np2(self, **kwargs):
        advanced_options_quantization_scale_type_default = self.runtime_options.get('advanced_options:quantization_scale_type', None)
        # do not modify the scale type if per channel quantization is supported in the device
        if advanced_options_quantization_scale_type_default != presets.QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN:
            kwargs['advanced_options:quantization_scale_type'] = presets.QUANTScaleType.QUANT_SCALE_TYPE_NP2
        #
        return self.get_runtime_options(presets.MODEL_TYPE_TFLITE, is_qat=False, **kwargs)

    def runtime_options_mxnet_np2(self, **kwargs):
        advanced_options_quantization_scale_type_default = self.runtime_options.get('advanced_options:quantization_scale_type', None)
        # do not modify the scale type if per channel quantization is supported in the device
        if advanced_options_quantization_scale_type_default != presets.QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN:
            kwargs['advanced_options:quantization_scale_type'] = presets.QUANTScaleType.QUANT_SCALE_TYPE_NP2
        #
        return self.get_runtime_options(presets.MODEL_TYPE_MXNET, is_qat=False, **kwargs)

    def runtime_options_onnx_p2(self, **kwargs):
        advanced_options_quantization_scale_type_default = self.runtime_options.get('advanced_options:quantization_scale_type', None)
        # do not modify the scale type if per channel quantization is supported in the device
        if advanced_options_quantization_scale_type_default != presets.QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN:
            kwargs['advanced_options:quantization_scale_type'] = presets.QUANTScaleType.QUANT_SCALE_TYPE_P2
        #
        return self.get_runtime_options(presets.MODEL_TYPE_ONNX, is_qat=False, **kwargs)

    def runtime_options_tflite_p2(self, **kwargs):
        advanced_options_quantization_scale_type_default = self.runtime_options.get('advanced_options:quantization_scale_type', None)
        # do not modify the scale type if per channel quantization is supported in the device
        if advanced_options_quantization_scale_type_default != presets.QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN:
            kwargs['advanced_options:quantization_scale_type'] = presets.QUANTScaleType.QUANT_SCALE_TYPE_P2
        #
        return self.get_runtime_options(presets.MODEL_TYPE_TFLITE, is_qat=False, **kwargs)

    def runtime_options_mxnet_p2(self, **kwargs):
        advanced_options_quantization_scale_type_default = self.runtime_options.get('advanced_options:quantization_scale_type', None)
        # do not modify the scale type if per channel quantization is supported in the device
        if advanced_options_quantization_scale_type_default != presets.QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN:
            kwargs['advanced_options:quantization_scale_type'] = presets.QUANTScaleType.QUANT_SCALE_TYPE_P2
        #
        return self.get_runtime_options(presets.MODEL_TYPE_MXNET, is_qat=False, **kwargs)

    def runtime_options_onnx_qat_v1(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = presets.QUANTScaleType.QUANT_SCALE_TYPE_P2
        # kwargs['advanced_options:prequantized_model'] = presets.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_CLIP
        return self.get_runtime_options(presets.MODEL_TYPE_ONNX, is_qat=True, **kwargs)

    def runtime_options_tflite_qat_v1(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = presets.QUANTScaleType.QUANT_SCALE_TYPE_P2
        # kwargs['advanced_options:prequantized_model'] = presets.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_CLIP
        return self.get_runtime_options(presets.MODEL_TYPE_TFLITE, is_qat=True, **kwargs)

    def runtime_options_mxnet_qat_v1(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = presets.QUANTScaleType.QUANT_SCALE_TYPE_P2
        # kwargs['advanced_options:prequantized_model'] = presets.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_CLIP
        return self.get_runtime_options(presets.MODEL_TYPE_MXNET, is_qat=True, **kwargs)

    def runtime_options_onnx_qat_v2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = presets.QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN
        kwargs['advanced_options:prequantized_model'] = presets.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_QDQ
        return self.get_runtime_options(presets.MODEL_TYPE_ONNX, is_qat=True, **kwargs)

    def runtime_options_onnx_qat_v2_p2(self, **kwargs):
        kwargs['advanced_options:quantization_scale_type'] = presets.QUANTScaleType.QUANT_SCALE_TYPE_P2
        kwargs['advanced_options:prequantized_model'] = presets.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_QDQ
        return self.get_runtime_options(presets.MODEL_TYPE_ONNX, is_qat=True, **kwargs)
