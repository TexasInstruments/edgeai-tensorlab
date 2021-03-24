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

        # quantization params
        self.quantization_params = QuantizationParams(self.tidl_tensor_bits, self.max_frames_calib,
                                                 self.max_calib_iterations, is_qat=False)
        self.quantization_params_qat = QuantizationParams(self.tidl_tensor_bits, self.max_frames_calib,
                                                 self.max_calib_iterations, is_qat=True)

    def get_session_name_to_cfg_dict(self, is_qat):
        quantization_params = self.quantization_params_qat if is_qat else self.quantization_params
        session_name_to_cfg_dict = dict()
        session_name_to_cfg_dict[constants.SESSION_NAME_TVMDLR] = quantization_params.get_session_tvmdlr_cfg()
        session_name_to_cfg_dict[constants.SESSION_NAME_TFLITERT] = quantization_params.get_session_tflitert_cfg()
        session_name_to_cfg_dict[constants.SESSION_NAME_ONNXRT] = quantization_params.get_session_onnxrt_cfg()
        return session_name_to_cfg_dict

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

    def get_session_cfg(self, model_type_or_session_name, is_qat):
        session_name = self.get_session_name(model_type_or_session_name)
        return self.get_session_name_to_cfg_dict(is_qat)[session_name]

    def get_session_type(self, model_type_or_session_name):
        session_name = self.get_session_name(model_type_or_session_name)
        return sessions.get_session_name_to_type_dict()[session_name]

    ###############################################################
    # preprocess transforms
    ###############################################################
    def _get_preproc_base(self, resize, crop, data_layout, reverse_channels,
                         backend, interpolation, resize_with_pad, mean, scale):
        transforms_list = [
            preprocess.ImageRead(backend=backend),
            preprocess.ImageResize(resize, interpolation=interpolation, resize_with_pad=resize_with_pad),
            preprocess.ImageCenterCrop(crop),
            preprocess.ImageToNPTensor4D(data_layout=data_layout),
            preprocess.ImageNormMeanScale(mean=mean, scale=scale, data_layout=data_layout)]
        if reverse_channels:
            transforms_list = transforms_list + [preprocess.NPTensor4DChanReverse(data_layout=data_layout)]
        #
        transforms = utils.TransformsCompose(transforms_list, resize=resize, crop=crop,
                                             data_layout=data_layout, reverse_channels=reverse_channels,
                                             backend=backend, interpolation=interpolation,
                                             mean=mean, scale=scale)
        return transforms

    def get_preproc_onnx(self, resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
                         backend='pil', interpolation=None, resize_with_pad=False,
                         mean=(123.675, 116.28, 103.53), scale=(0.017125, 0.017507, 0.017429)):
        transforms = self._get_preproc_base(resize=resize, crop=crop, data_layout=data_layout,
                                      reverse_channels=reverse_channels, backend=backend, interpolation=interpolation,
                                      resize_with_pad=resize_with_pad, mean=mean, scale=scale)
        return transforms

    def get_preproc_jai(self, resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False,
                        mean=(128.0, 128.0, 128.0), scale=(1/64.0, 1/64.0, 1/64.0)):
        return self._get_preproc_base(resize=resize, crop=crop, data_layout=data_layout, reverse_channels=reverse_channels,
                                backend=backend, interpolation=interpolation, resize_with_pad=resize_with_pad,
                                mean=mean, scale=scale)

    def get_preproc_mxnet(self, resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=None, resize_with_pad=False,
                        mean=(123.675, 116.28, 103.53), scale=(0.017125, 0.017507, 0.017429)):
        return self._get_preproc_base(resize=resize, crop=crop, data_layout=data_layout, reverse_channels=reverse_channels,
                                backend=backend, interpolation=interpolation, resize_with_pad=resize_with_pad,
                                mean=mean, scale=scale)

    def get_preproc_tflite(self, resize=256, crop=224, data_layout=constants.NHWC, reverse_channels=False,
                              backend='pil', interpolation=None, resize_with_pad=False,
                              mean=(128.0, 128.0, 128.0), scale=(1/128.0, 1/128.0, 1/128.0)):
        return self._get_preproc_base(resize=resize, crop=crop, data_layout=data_layout, reverse_channels=reverse_channels,
                                backend=backend, interpolation=interpolation, resize_with_pad=resize_with_pad,
                                mean=mean, scale=scale)

    def get_postproc_classification(self):
        postprocess_classification = [postprocess.IndexArray(), postprocess.ArgMax()]
        transforms = utils.TransformsCompose(postprocess_classification)
        return transforms

    ###############################################################
    # post process transforms for detection
    ###############################################################
    def _get_postproc_detection_base(self, formatter=None, resize_with_pad=False, normalized_detections=True,
                                     shuffle_indices=None, squeeze_axis=0):
        postprocess_detection = [postprocess.ShuffleList(indices=shuffle_indices),
                                 postprocess.Concat(axis=-1, end_index=3)]
        if squeeze_axis is not None:
            #  TODO make this more generic to squeeze any axis
            postprocess_detection += [postprocess.IndexArray()]
        #
        if formatter is not None:
            postprocess_detection += [formatter]
        #
        postprocess_detection += [postprocess.DetectionResizePad(resize_with_pad=resize_with_pad,
                                                    normalized_detections=normalized_detections)]
        if self.detection_thr is not None:
            postprocess_detection += [postprocess.DetectionFilter(detection_thr=self.detection_thr,
                                                                  detection_max=self.detection_max)]
        #
        if self.save_output:
            postprocess_detection += [postprocess.DetectionImageSave()]
        #
        transforms = utils.TransformsCompose(postprocess_detection, detection_thr=self.detection_thr,
                                             save_output=self.save_output, formatter=formatter)
        return transforms

    def get_postproc_detection_onnx(self, formatter=None, **kwargs):
        return self._get_postproc_detection_base(formatter=formatter, **kwargs)

    def get_postproc_detection_tflite(self, formatter=postprocess.DetectionYXYX2XYXY(), **kwargs):
        return self._get_postproc_detection_base(formatter=formatter, **kwargs)

    def get_postproc_detection_mxnet(self, formatter=None, resize_with_pad=False,
                        normalized_detections=False, shuffle_indices=(2,0,1), **kwargs):
        return self._get_postproc_detection_base(formatter=formatter, resize_with_pad=resize_with_pad,
                        normalized_detections=normalized_detections, shuffle_indices=shuffle_indices, **kwargs)

    ###############################################################
    # post process transforms for segmentation
    ###############################################################
    def _get_postproc_segmentation_base(self, data_layout, with_argmax=True):
        channel_axis = -1 if data_layout == constants.NHWC else 1
        postprocess_segmentation = [postprocess.IndexArray()]
        if with_argmax:
            postprocess_segmentation += [postprocess.ArgMax(axis=channel_axis)]
        #
        postprocess_segmentation += [postprocess.NPTensorToImage(data_layout=data_layout),
                                     postprocess.SegmentationImageResize()]
        if self.save_output:
            postprocess_segmentation += [postprocess.SegmentationImageSave()]
        #
        transforms = utils.TransformsCompose(postprocess_segmentation, data_layout=data_layout,
                                             save_output=self.save_output, with_argmax=with_argmax)
        return transforms

    def get_postproc_segmentation_onnx(self, data_layout=constants.NCHW, with_argmax=True):
        return self._get_postproc_segmentation_base(data_layout=data_layout, with_argmax=with_argmax)

    def get_postproc_segmentation_tflite(self, data_layout=constants.NHWC, with_argmax=True):
        return self._get_postproc_segmentation_base(data_layout=data_layout, with_argmax=with_argmax)


############################################################
# quantization / calibration params
class QuantizationParams():
    def __init__(self, tidl_tensor_bits, max_frames_calib, max_calib_iterations, is_qat=False):
        self.tidl_tensor_bits = tidl_tensor_bits
        self.max_frames_calib = max_frames_calib
        self.max_calib_iterations = max_calib_iterations
        self.is_qat = is_qat

    def get_num_frames_calib(self):
        return self.max_frames_calib if self.tidl_tensor_bits == 8 else 1

    def get_num_calib_iterations(self):
        return self.max_calib_iterations if self.tidl_tensor_bits == 8 else 1

    def get_tidl_calibration_accuracy_level(self):
        return 0 if self.tidl_tensor_bits != 8 or self.is_qat else 1

    def get_tidl_basic_options(self):
        tidl_basic_options = {
            'tidl_tensor_bits': self.tidl_tensor_bits,
            'accuracy_level': self.get_tidl_calibration_accuracy_level(),
        }
        return tidl_basic_options

    def get_tidl_advanced_options(self):
        tidl_advanced_options = {
            'calibration_frames': self.get_num_frames_calib(),
            'calibration_iterations': self.get_num_calib_iterations(),
            'quantization_scale_type': 1 if self.is_qat else 0
        }
        return tidl_advanced_options

    def get_session_tvmdlr_cfg(self):
        runtime_options = self.get_tidl_basic_options()
        runtime_options.update({'advanced_options':self.get_tidl_advanced_options()})
        session_tvmdlr_cfg = {'runtime_options': runtime_options}
        return session_tvmdlr_cfg

    def get_session_tflitert_cfg(self):
        runtime_options = self.get_tidl_basic_options()
        runtime_options.update({'advanced_options:'+k:v for k,v in self.get_tidl_advanced_options().items()})
        session_tflitert_cfg = {'runtime_options': runtime_options}
        return session_tflitert_cfg

    def get_session_onnxrt_cfg(self):
        runtime_options = self.get_tidl_basic_options()
        runtime_options.update({'advanced_options:'+k:v for k,v in self.get_tidl_advanced_options().items()})
        session_onnxrt_cfg = {'runtime_options': runtime_options}
        return session_onnxrt_cfg


