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
        # # yolov3: detection - yolov3 416x416 - expected_metric: 31.0% COCO AP[0.5-0.95]
        # 'vdet-12-020-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_onnx((416,416), (416,416), backend='cv2',
        #         mean=(0.0, 0.0, 0.0), scale=(1/255.0, 1/255.0, 1/255.0)),
        #     session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx,
        #         model_path=f'{settings.models_path}/vision/detection/coco/onnx-models/yolov3-10.onnx',
        #         input_shape=dict(input_1=(1,3,416,416), image_shape=(1,2)),
        #         extra_inputs=dict(image_shape=np.array([416,416], dtype=np.float32)[np.newaxis,...])),
        #     postprocess=postproc_detection_onnx,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':31.0})
        # ),
        # # jai-devkit/pytorch-mmdetection: detection - ssd-lite_regnetx-800mf_fpn_bgr_512x512 - expected_metric: 32.8% COCO AP[0.5-0.95]
        # 'vdet-12-110-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_onnx((512,512), (512,512), backend='cv2'),
        #     session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx,
        #         model_path=f'{settings.models_path}/vision/detection/coco/ti-edgeai/ssd-lite_regnetx-800mf_fpn_bgr_512x512_20200919_model.onnx'),
        #     postprocess=settings.get_postproc_detection_onnx(squeeze_axis=None, normalized_detections=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':32.8})
        # ),
        #################################################################
        #       MXNET MODELS
        #################################################################
        #################################################################
        #       TFLITE MODELS
        #################tflite models###################################
        # # tensorflow1.0 models: detection - ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18 expected_metric: 26.6% ap[0.5:0.95] accuracy
        # 'vdet-12-403-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_tflite((320,320), (320,320), backend='cv2'),
        #     session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite,
        #         model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18.tflite'),
        #     postprocess=postproc_detection_tflite,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':26.6})
        # ),
        # # tensorflow1.0 models: detection - ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03 expected_metric: 32.0% ap[0.5:0.95] accuracy
        # 'vdet-12-404-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_tflite((640,640), (640,640), backend='cv2'),
        #     session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite,
        #         model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tflite'),
        #     postprocess=postproc_detection_tflite,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':32.0})
        # ),
        #################################################################
        # tensorflow2.0 models: detection - ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8 expected_metric: 28.2% ap[0.5:0.95] accuracy
        # 'vdet-12-450-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_tflite((640,640), (640,640), backend='cv2'),
        #     session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite,
        #         model_path=f'{settings.models_path}/vision/detection/coco/tf2-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tflite'),
        #     postprocess=postproc_detection_tflite,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':28.2})
        # ),
        # # tensorflow2.0 models: detection - ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 expected_metric: 34.3% ap[0.5:0.95] accuracy
        # 'vdet-12-451-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_tflite((640,640), (640,640), backend='cv2'),
        #     session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite,
        #         model_path=f'{settings.models_path}/vision/detection/coco/tf2-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tflite'),
        #     postprocess=postproc_detection_tflite,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':34.3})
        # ),
        # # tensorflow2.0 models: detection - ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8 expected_metric: 38.3% ap[0.5:0.95] accuracy
        # 'vdet-12-452-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_tflite((1024,1024), (1024,1024), backend='cv2'),
        #     session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite,
        #         model_path=f'{settings.models_path}/vision/detection/coco/tf2-models/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tflite'),
        #     postprocess=postproc_detection_tflite,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':38.3})
        # ),
        #################################################################
        # # google automl: detection - efficientdet-lite0_bifpn_maxpool2x2_relu expected_metric: 33.5% ap[0.5:0.95] accuracy
        # 'vdet-12-040-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_tflite((512,512), (512,512), backend='cv2'),
        #     session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite,
        #         model_path=f'{settings.models_path}/vision/detection/coco/google-automl/efficientdet-lite0_bifpn_maxpool2x2_relu_ti-lite.tflite'),
        #     postprocess=postproc_detection_tflite,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':33.5})
        # ),
    }
    return pipeline_configs

