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
from edgeai_benchmark import constants, utils, datasets, preprocess, sessions, postprocess, metrics


def get_configs(settings, work_dir):
    # get the sessions types to use for each model type
    onnx_session_type = settings.get_session_type(constants.MODEL_TYPE_ONNX)
    tflite_session_type = settings.get_session_type(constants.MODEL_TYPE_TFLITE)
    mxnet_session_type = settings.get_session_type(constants.MODEL_TYPE_MXNET)

    preproc_transforms = preprocess.PreProcessTransforms(settings)
    postproc_transforms = postprocess.PostProcessTransforms(settings)

    # configs for each model pipeline
    common_cfg = {
        'task_type': 'detection',
        'dataset_category': datasets.DATASET_CATEGORY_COCO,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_COCO]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_COCO]['input_dataset'],
    }

    postproc_detection_onnx = postproc_transforms.get_transform_detection_onnx()
    postproc_detection_tflite = postproc_transforms.get_transform_detection_tflite()
    postproc_detection_efficientdet_ti_lite_tflite = postproc_transforms.get_transform_detection_tflite(normalized_detections=False, ignore_index=0,
                                                            formatter=postprocess.DetectionFormatting(dst_indices=(0,1,2,3,4,5), src_indices=(1,0,3,2,5,4)),
                                                            )
    postproc_detection_mxnet = postproc_transforms.get_transform_detection_mxnet()

    # reduce these iterations for slow models
    calibration_frames_fast = min(10, settings.calibration_frames)
    calibration_iterations_fast = min(5, settings.calibration_iterations)

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################onnx models#####################################
        #yolov4_scaled
        'od-8800':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640,640,resize_with_pad=True, backend='cv2'),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir ,  input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627) ),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={
                    'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list':f'../edgeai-modelforest/models-cl/vision/detection/coco/scaled-yolov4-gplv3/scaled-yolov4-csp_lite_640x640_20240220_model.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'/module_list.143/Conv2d/Conv_output_0'
                     }
                     ),
                model_path=f'../edgeai-modelforest/models-cl/vision/detection/coco/scaled-yolov4-gplv3/scaled-yolov4-csp_lite_640x640_20240220_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True,formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':45.8}, model_shortlist=70, compact_name='scaled-yolov4-csp-lite-640x640-gplv3', shortlisted=False)
        ),
        #yolov5-nano
         'od-8810':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640,  resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                     'object_detection:meta_layers_names_list':f'../edgeai-modelforest/models-cl/vision/detection/coco/edgeai-mmyolo-gplv3/yolov5_nano-v61_lite_640x640_20240329_model.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'1,3,142,150,158'},
                     ),
                model_path=f'../edgeai-modelforest/models-cl/vision/detection/coco/edgeai-mmyolo-gplv3/yolov5_nano-v61_lite_640x640_20240329_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 25.1}, model_shortlist=70, compact_name='yolov5-nano-v61-lite-640x640-gplv3', shortlisted=False)
        ),
        # yolov5-small
        'od-8820':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640,  resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                     'object_detection:meta_layers_names_list':f'../edgeai-modelforest/models-cl/vision/detection/coco/edgeai-mmyolo-gplv3/yolov5_small-v61_lite_640x640_20240329_model.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'1,3,142,150,158'}),
                model_path=f'../edgeai-modelforest/models-cl/vision/detection/coco/edgeai-mmyolo-gplv3/yolov5_small-v61_lite_640x640_20240329_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 35.5}, model_shortlist=70, compact_name='yolov5-small-v61-lite-640x640-gplv3', shortlisted=False)
        ),
        #yolov6_n mmyolo
        # 'od-8840':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx(640, 640,  resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
        #     session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
        #         runtime_options=settings.runtime_options_onnx_np2(
        #             det_options=True, ext_options={'object_detection:meta_arch_type': 6,
        #              'object_detection:meta_layers_names_list':f'/data/files/a0508577/work/edgeai-algo/edgeai-mmyolo-gplv3/work_dirs/yolov6_n_syncbn_fast_8xb32-400e_coco/epoch_400.prototxt',
        #             #  'advanced_options:output_feature_16bit_names_list':'1,142,150,158'
        #              }),
        #         model_path=f'/data/files/a0508577/work/edgeai-algo/edgeai-mmyolo-gplv3/work_dirs/yolov6_n_syncbn_fast_8xb32-400e_coco/epoch_400.onnx'),
        #     postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 35.3}, model_shortlist=None, compact_name='yolov6-n-lite-640x640-gplv3', shortlisted=False)
        # ),
        #yolov7-tiny
        'od-8850':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640,  resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                     'object_detection:meta_layers_names_list':f'../edgeai-modelforest/models-cl/vision/detection/coco/edgeai-mmyolo-gplv3/yolov7_tiny_lite_640x640_20230830_model.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'1,137,147,157'}),
                model_path=f'../edgeai-modelforest/models-cl/vision/detection/coco/edgeai-mmyolo-gplv3/yolov7_tiny_lite_640x640_20230830_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':36.7}, model_shortlist=70, compact_name='yolov7-tiny-lite-640x640-gplv3', shortlisted=False)
        ),
        #yolov7-large
        'od-8860':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, resize_with_pad=True, backend='cv2',pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                     'object_detection:meta_layers_names_list':f'../edgeai-modelforest/models-cl/vision/detection/coco/edgeai-mmyolo-gplv3/yolov7_large_lite_640x640_20240119_model.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'2,221,251,277'
                     }),
                model_path=f'../edgeai-modelforest/models-cl/vision/detection/coco/edgeai-mmyolo-gplv3/yolov7_large_lite_640x640_20240119_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':48.1}, model_shortlist=70, compact_name='yolov7-large-lite-640x640-gplv3', shortlisted=False)
        ),
        #yolov8-nano
        'od-8870':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640,  resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                     det_options=True, ext_options={'object_detection:meta_arch_type': 8,
                     'object_detection:meta_layers_names_list':f'../edgeai-modelforest/models-cl/vision/detection/coco/edgeai-mmyolo-gplv3/yolov8_nano_lite_640x640_20231118_model.prototxt',
                     'advanced_options:output_feature_16bit_names_list':''
                     }),
                model_path=f'../edgeai-modelforest/models-cl/vision/detection/coco/edgeai-mmyolo-gplv3/yolov8_nano_lite_640x640_20231118_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':34.5}, model_shortlist=70, compact_name='yolov8-nano-lite-640x640-gplv3', shortlisted=False)
        ),
        #yolov8-small
        'od-8880':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640,  resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                     det_options=True, ext_options={'object_detection:meta_arch_type': 8,
                     'object_detection:meta_layers_names_list':f'../edgeai-modelforest/models-cl/vision/detection/coco/edgeai-mmyolo-gplv3/yolov8_small_lite_640x640_20231117_model.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'1,129,134,139,144,149,154'
                     }),
                model_path=f'../edgeai-modelforest/models-cl/vision/detection/coco/edgeai-mmyolo-gplv3/yolov8_small_lite_640x640_20231117_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':42.4}, model_shortlist=70, compact_name='yolov8-small-lite-640x640-gplv3', shortlisted=False)
        ),
        # yolox_tiny lite versions from mmyolo
        'od-8890':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(416, 416, reverse_channels=True, resize_with_pad=[True,'corner'], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': f'../edgeai-modelforest/models-cl/vision/detection/coco/edgeai-mmyolo-gplv3/yolox_tiny_lite_416x416_20231127_model.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '1,162,163,164,173,174,175,184,185,186'
                    }),
                model_path=f'../edgeai-modelforest/models-cl/vision/detection/coco/edgeai-mmyolo-gplv3/yolox_tiny_lite_416x416_20231127_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=[True,'corner'], formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 31.1}, model_shortlist=70, compact_name='yolox-tiny-lite-416x416-gplv3', shortlisted=False)
        ),
        # yolox_small lite versions from mmyolo
        'od-8900':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=True, resize_with_pad=[True,'corner'], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': f'../edgeai-modelforest/models-cl/vision/detection/coco/edgeai-mmyolo-gplv3/yolox_small_lite_640x640_20240319_model.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '1,162,163,164,173,174,175,184,185,186'
                    }),
                model_path=f'../edgeai-modelforest/models-cl/vision/detection/coco/edgeai-mmyolo-gplv3/yolox_small_lite_640x640_20240319_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=[True,'corner'], formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 31.1}, model_shortlist=70, compact_name='yolox-small-lite-640x640-gplv3', shortlisted=False)
        ),
    }
    return pipeline_configs

