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

import numpy as np
from jai_benchmark import constants, utils, datasets, preprocess, sessions, postprocess, metrics


def get_configs(settings, work_dir):
    # get the sessions types to use for each model type
    onnx_session_type = settings.get_session_type(constants.MODEL_TYPE_ONNX)
    tflite_session_type = settings.get_session_type(constants.MODEL_TYPE_TFLITE)
    mxnet_session_type = settings.get_session_type(constants.MODEL_TYPE_MXNET)

    # for onnx and mxnet float models, we set non-power-of-2 scale for quant here - optional
    runtime_options_onnx = settings.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=False,
                                    runtime_options={'advanced_options:quantization_scale_type': 0})
    runtime_options_tflite = settings.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=False)
    runtime_options_mxnet = settings.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=False,
                                    runtime_options={'advanced_options:quantization_scale_type': 0})

    runtime_options_onnx_qat = settings.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=True)
    runtime_options_tflite_qat = settings.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=True)
    runtime_options_mxnet_qat = settings.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=True)

    # configs for each model pipeline
    common_cfg = {
        'task_type': 'detection',
        'calibration_dataset': settings.dataset_cache['coco']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['coco']['input_dataset'],
    }

    common_session_cfg = dict(work_dir=work_dir, target_device=settings.target_device)

    postproc_detection_onnx = settings.get_postproc_detection_onnx()
    postproc_detection_tflite = settings.get_postproc_detection_tflite()
    postproc_detection_mxnet = settings.get_postproc_detection_mxnet()

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################onnx models#####################################
        #################################################################
        #       MXNET MODELS
        #################################################################
        # mxnet : gluoncv model : detection - yolo3_mobilenet1.0_coco - accuracy: 28.6% ap[0.5:0.95], 48.9% ap50
        'vdet-12-060-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((416,416), (416,416), backend='cv2'),
            session=mxnet_session_type(**common_session_cfg, runtime_options=runtime_options_mxnet,
                model_path=[f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/yolo3_mobilenet1.0_coco-symbol.json',
                            f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/yolo3_mobilenet1.0_coco-0000.params'],
                model_type='mxnet', input_shape={'data':(1,3,416,416)}),
            postprocess=postproc_detection_mxnet,
            metric=dict(label_offset_pred=coco_label_offset_80to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':28.6})
        ),
        # mxnet : gluoncv model : detection - ssd_512_mobilenet1.0_coco - accuracy: 21.7% ap[0.5:0.95], 39.2% ap50
        'vdet-12-061-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((512,512), (512,512), backend='cv2'),
            session=mxnet_session_type(**common_session_cfg, runtime_options=runtime_options_mxnet,
                model_path=[f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_mobilenet1.0_coco-symbol.json',
                            f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_mobilenet1.0_coco-0000.params'],
                model_type='mxnet', input_shape={'data':(1,3,512,512)}),
            postprocess=postproc_detection_mxnet,
            metric=dict(label_offset_pred=coco_label_offset_80to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':21.7})
        ),
        #################################################################
        #       TFLITE MODELS
        #################tflite models###################################
        # mlperf edge: detection - ssd_mobilenet_v1_coco_2018_01_28 expected_metric: 23.0% ap[0.5:0.95] accuracy
        'vdet-12-010-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((300,300), (300,300), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite,
                model_path=f'{settings.models_path}/vision/detection/coco/mlperf/ssd_mobilenet_v1_coco_2018_01_28.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=coco_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':23.0})
        ),
        # mlperf mobile: detection - ssd_mobilenet_v2_coco_300x300 - expected_metric: 22.0% COCO AP[0.5-0.95]
        'vdet-12-011-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((300,300), (300,300), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite,
                model_path=f'{settings.models_path}/vision/detection/coco/mlperf/ssd_mobilenet_v2_300_float.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=coco_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':22.0})
        ),
        #################################################################
        # tensorflow1.0 models: detection - ssdlite_mobiledet_dsp_320x320_coco_2020_05_19 expected_metric: 28.9% ap[0.5:0.95] accuracy
        'vdet-12-400-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((320,320), (320,320), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite,
                model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_2020_05_19.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=coco_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':28.9})
        ),
        # tensorflow1.0 models: detection - ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19 expected_metric: 25.9% ap[0.5:0.95] accuracy
        'vdet-12-401-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((320,320), (320,320), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite,
                model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=coco_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':25.9})
        ),
        # tensorflow1.0 models: detection - ssdlite_mobilenet_v2_coco_2018_05_09 expected_metric: 22.0% ap[0.5:0.95] accuracy
        'vdet-12-402-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((300,300), (300,300), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite,
                model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssdlite_mobilenet_v2_coco_2018_05_09.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=coco_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':22.0})
        ),
    }
    return pipeline_configs



################################################################################################
# convert from 80 class index (typical output of a mmdetection detector) to 90 or 91 class
# (original labels of coco starts from 1, and 0 is background)
# the evalation/metric script will convert from 80 class to coco's 90 class.
def coco_label_offset_80to90(label_offset=1):
    coco_label_table = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                         21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                         41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                         61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                         81, 82, 84, 85, 86, 87, 88, 89, 90]

    if label_offset == 1:
        # 0 => 1, 1 => 2, .. 79 -> 90, 80 => 91
        coco_label_offset = {k:v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({80:91})
    elif label_offset == 0:
        # 0 => 0, 1 => 1, .. 80 => 90
        coco_label_offset = {(k+1):v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({0:0})
    else:
        assert False, f'unsupported value for label_offset {label_offset}'
    #
    return coco_label_offset


# convert from 90 class index (typical output of a tensorflow detector) to 90 or 91 class
# (original labels of coco starts from 1, and 0 is background)
def coco_label_offset_90to90(label_offset=1):
    coco_label_table = range(1,91)
    if label_offset == 1:
        # 0 => 1, 1 => 2, .. 90 => 91
        coco_label_offset = {k:v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({-1:0,90:91})
    elif label_offset == 0:
        # 0 => 0, 1 => 1, .. 90 => 90
        coco_label_offset = {(k+1):v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({-1:-1,0:0})
    else:
        assert False, f'unsupported value for label_offset {label_offset}'
    #
    return coco_label_offset