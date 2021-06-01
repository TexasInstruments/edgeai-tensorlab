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

    # Default for ONNX/MXNET Models: Non-Power-2, TFLITE-Power2
    # For selected model we toggle based on which ever is better from accuracy perspective
    runtime_options_onnx_np2 = settings.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=False,
                                    runtime_options={'advanced_options:quantization_scale_type': 0})
    runtime_options_tflite_np2 = settings.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=False,
                                    runtime_options={'advanced_options:quantization_scale_type': 0})
    runtime_options_mxnet_np2 = settings.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=False,
                                    runtime_options={'advanced_options:quantization_scale_type': 0})

    runtime_options_onnx_p2 = settings.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=False,)
    runtime_options_tflite_p2 = settings.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=False,)
    runtime_options_mxnet_p2 = settings.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=False,)

    #This option should go away after testing
    use_default_power_2_setting = False
    if use_default_power_2_setting:
        runtime_options_onnx_p2 = runtime_options_onnx_np2
        runtime_options_tflite_np2 = runtime_options_tflite_p2
        runtime_options_mxnet_p2 = runtime_options_mxnet_np2

    runtime_options_onnx_qat = settings.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=True)
    runtime_options_tflite_qat = settings.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=True)
    runtime_options_mxnet_qat = settings.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=True)

    runtime_options_onnx_ssd = settings.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=False,
                                    runtime_options={'advanced_options:quantization_scale_type': 0,
                                                     'deny_list': "Reshape"})

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
        # mlperf edge: detection - coco_ssd-resnet34_1200x1200 - expected_metric: 20.0% COCO AP[0.5-0.95]
        'vdet-12-012-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((1200,1200), (1200,1200), backend='cv2'),
            session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx_ssd,
                model_path=f'{settings.models_path}/vision/detection/coco/mlperf/ssd_resnet34-ssd1200.onnx'),
            postprocess=postproc_detection_onnx,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=0)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':20.0})
        ),
        #################################################################
        #       MXNET MODELS
        #################################################################
        # mxnet : gluoncv model : detection - yolo3_mobilenet1.0_coco - accuracy: 28.6% ap[0.5:0.95], 48.9% ap50
        'vdet-12-060-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((416,416), (416,416), backend='cv2'),
            session=mxnet_session_type(**common_session_cfg, runtime_options=runtime_options_mxnet_np2,
                model_path=[f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/yolo3_mobilenet1.0_coco-symbol.json',
                            f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/yolo3_mobilenet1.0_coco-0000.params'],
                model_type='mxnet', input_shape={'data':(1,3,416,416)}),
            postprocess=postproc_detection_mxnet,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':28.6})
        ),
        # mxnet : gluoncv model : detection - ssd_512_mobilenet1.0_coco - accuracy: 21.7% ap[0.5:0.95], 39.2% ap50
        'vdet-12-061-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((512,512), (512,512), backend='cv2'),
            session=mxnet_session_type(**common_session_cfg, runtime_options=runtime_options_mxnet_np2,
                model_path=[f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_mobilenet1.0_coco-symbol.json',
                            f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_mobilenet1.0_coco-0000.params'],
                model_type='mxnet', input_shape={'data':(1,3,512,512)}),
            postprocess=postproc_detection_mxnet,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':21.7})
        ),
        # mxnet : gluoncv model : detection - ssd_512_resnet50_v1_coco - accuracy: 30.6% ap[0.5:0.95], 50.0% ap50
        'vdet-12-062-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((512,512), (512,512), backend='cv2'),
            session=mxnet_session_type(**common_session_cfg, runtime_options=runtime_options_mxnet_p2,
                model_path=[f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_resnet50_v1_coco-symbol.json',
                            f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_resnet50_v1_coco-0000.params'],
                model_type='mxnet', input_shape={'data':(1,3,512,512)}),
            postprocess=postproc_detection_mxnet,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':30.6})
        ),
        # mxnet : gluoncv model : detection - yolo3_darknet53_coco - accuracy: 36.0% ap[0.5:0.95], 57.2% ap50
        'vdet-12-063-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((416,416), (416,416), backend='cv2'),
            session=mxnet_session_type(**common_session_cfg, runtime_options=runtime_options_mxnet_p2,
                model_path=[f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/yolo3_darknet53_coco-symbol.json',
                            f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/yolo3_darknet53_coco-0000.params'],
                model_type='mxnet', input_shape={'data':(1,3,416,416)}),
            postprocess=postproc_detection_mxnet,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':36.0})
        ),
        #################################################################
        #       TFLITE MODELS
        #################tflite models###################################
        # mlperf edge: detection - ssd_mobilenet_v1_coco_2018_01_28 expected_metric: 23.0% ap[0.5:0.95] accuracy
        'vdet-12-010-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((300,300), (300,300), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_p2,
                model_path=f'{settings.models_path}/vision/detection/coco/mlperf/ssd_mobilenet_v1_coco_2018_01_28.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':23.0})
        ),
        # mlperf mobile: detection - ssd_mobilenet_v2_coco_300x300 - expected_metric: 22.0% COCO AP[0.5-0.95]
        'vdet-12-011-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((300,300), (300,300), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_p2,
                model_path=f'{settings.models_path}/vision/detection/coco/mlperf/ssd_mobilenet_v2_300_float.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':22.0})
        ),
        #################################################################
        # tensorflow1.0 models: detection - ssdlite_mobiledet_dsp_320x320_coco_2020_05_19 expected_metric: 28.9% ap[0.5:0.95] accuracy
        'vdet-12-400-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((320,320), (320,320), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_np2,
                model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_2020_05_19.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':28.9})
        ),
        # tensorflow1.0 models: detection - ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19 expected_metric: 25.9% ap[0.5:0.95] accuracy
        'vdet-12-401-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((320,320), (320,320), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_p2,
                model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':25.9})
        ),
        # tensorflow1.0 models: detection - ssdlite_mobilenet_v2_coco_2018_05_09 expected_metric: 22.0% ap[0.5:0.95] accuracy
        'vdet-12-402-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((300,300), (300,300), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_p2,
                model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssdlite_mobilenet_v2_coco_2018_05_09.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':22.0})
        ),
    }
    return pipeline_configs

