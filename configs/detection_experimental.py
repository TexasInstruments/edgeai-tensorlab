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

    # use a large top_k, keep_top_k and low confidence_threshold for accuracy measurement
    runtime_options_tflite_np2 = settings.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=False,
                runtime_options={'object_detection:confidence_threshold': settings.detection_thr,
                                 'object_detection:nms_threshold': 0.45,
                                 'object_detection:top_k': 500,
                                 #'object_detection:keep_top_k': 100
                                 })

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
        #     session=onnx_session_type(**common_session_cfg, runtime_options=settings.runtime_options_onnx_p2(),
        #         model_path=f'{settings.models_path}/vision/detection/coco/onnx-models/yolov3-10.onnx',
        #         input_shape=dict(input_1=(1,3,416,416), image_shape=(1,2)),
        #         extra_inputs=dict(image_shape=np.array([416,416], dtype=np.float32)[np.newaxis,...])),
        #     postprocess=postproc_detection_onnx,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':31.0})
        # ),
        'vdet-12-020-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx(640, 640,  resize_with_pad=True, mean=(0.0, 0.0, 0.0), scale=(0.003921568627, 0.003921568627, 0.003921568627), backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(),
                                    {'object_detection:meta_arch_type': 6,
                                     'object_detection:meta_layers_names_list':f'../edgeai-yolov5/pretrained_models/models/yolov5s6_640_ti_lite/weights/yolov5s6_640_ti_lite_metaarch.prototxt',
                                     'advanced_options:output_feature_16bit_names_list':'370, 680, 990, 1300'
                                     }),
                model_path=f'../edgeai-yolov5/pretrained_models/models/yolov5s6_640_ti_lite/weights/yolov5s6_640_ti_lite_37p4_56p0.onnx'),
            postprocess=settings.get_postproc_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':37.4})
        ),
        'vdet-12-021-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx(384, 384,  resize_with_pad=True, mean=(0.0, 0.0, 0.0), scale=(0.003921568627, 0.003921568627, 0.003921568627), backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(),
                                    {'object_detection:meta_arch_type': 6,
                                     'object_detection:meta_layers_names_list':f'../edgeai-yolov5/pretrained_models/models/yolov5s6_384_ti_lite/weights/yolov5s6_384_ti_lite_metaarch.prototxt',
                                     'advanced_options:output_feature_16bit_names_list':'168, 370, 680, 990, 1300'
                                     }),
                model_path=f'../edgeai-yolov5/pretrained_models/models/yolov5s6_384_ti_lite/weights/yolov5s6_384_ti_lite_32p8_51p2.onnx'),
            postprocess=settings.get_postproc_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':32.8})
        ),
        'vdet-12-022-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx(640, 640, resize_with_pad=True, mean=(0.0, 0.0, 0.0), scale=(0.003921568627, 0.003921568627, 0.003921568627), backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(),
                                    {'object_detection:meta_arch_type': 6,
                                     'object_detection:meta_layers_names_list':f'../edgeai-yolov5/pretrained_models/models/yolov5m6_640_ti_lite/weights/yolov5m6_640_ti_lite_metaarch.prototxt',
                                     'advanced_options:output_feature_16bit_names_list':'228, 498, 808, 1118, 1428'
                                     }),
                model_path=f'../edgeai-yolov5/pretrained_models/models/yolov5m6_640_ti_lite/weights/yolov5m6_640_ti_lite_44p1_62p9.onnx'),
            postprocess=settings.get_postproc_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False,  resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':44.1})
        ),
           'vdet-12-023-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx(640, 640, resize_with_pad=True, mean=(0.0, 0.0, 0.0), scale=(0.003921568627, 0.003921568627, 0.003921568627), backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(),
                                    {'object_detection:meta_arch_type': 6,
                                     'object_detection:meta_layers_names_list':f'../edgeai-yolov5/pretrained_models/models/yolov5l6_640_ti_lite/weights/yolov5l6_640_ti_lite_metaarch.prototxt',
                                     'advanced_options:output_feature_16bit_names_list':'288, 626, 936, 1246, 1556'
                                     }),
                model_path=f'../edgeai-yolov5/pretrained_models/models/yolov5l6_640_ti_lite/weights/yolov5l6_640_ti_lite_47p1_65p6.onnx'),
            postprocess=settings.get_postproc_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False,  resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':47.1})
        ),

        #################################################################
        #       MXNET MODELS
        #################################################################
        # # mxnet : gluoncv model : detection - yolo3_darknet53_coco - accuracy: 36.0% ap[0.5:0.95], 57.2% ap50
        # 'vdet-12-063-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_onnx((416,416), (416,416), backend='cv2'),
        #     session=mxnet_session_type(**common_session_cfg, runtime_options=settings.runtime_options_mxnet_p2(),
        #         model_path=[f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/yolo3_darknet53_coco-symbol.json',
        #                     f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/yolo3_darknet53_coco-0000.params'],
        #         model_type='mxnet', input_shape={'data':(1,3,416,416)}),
        #     postprocess=postproc_detection_mxnet,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':36.0})
        # ),
        # # mxnet : gluoncv model : detection - center_net_resnet18_v1b_coco - accuracy: 26.6% ap[0.5:0.95], 28.1% ap50
        # 'vdet-12-064-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_onnx((512,512), (512,512), backend='cv2'),
        #     session=mxnet_session_type(**common_session_cfg, runtime_options=settings.runtime_options_mxnet_p2(),
        #         model_path=[f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/center_net_resnet18_v1b_coco-symbol.json',
        #                     f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/center_net_resnet18_v1b_coco-0000.params'],
        #         model_type='mxnet', input_shape={'data':(1,3,512,512)}),
        #     postprocess=postproc_detection_mxnet,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':26.6})
        # ),
        #################################################################
        #       TFLITE MODELS
        #################tflite models###################################
        # 'vdet-12-415-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_tflite((1024,1024), (1024,1024), backend='cv2'),
        #     session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_np2,
        #         model_path=f'{settings.models_path}/vision/detection/coco/tf2-models/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tflite'),
        #     postprocess=postproc_detection_tflite,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':38.3})
        # ),
    }
    return pipeline_configs

