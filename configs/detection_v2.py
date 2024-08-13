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
import onnxruntime

# for transformer models we need to set graph_optimization_level = ORT_DISABLE_ALL for onnxruntime
ORT_DISABLE_ALL = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
ORT_ENABLE_EXTENDED = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

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
        # Transformer models from huggingface transformers
        'od-8920':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((800,800),(800,800), resize_with_pad=False, backend='cv2'),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={
                     'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL 
                     }),
                model_path=f'{settings.models_path}/vision/detection/coco/hf-transformers/detr_fb_resnet50_800x800_simp.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, reshape_list=[(-1,4),(-1,1),(-1,1)],logits_bbox_to_bbox_ls=True,formatter=postprocess.DetectionXYWH2XYXYCenterXY()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90(label_offset=0,num_classes=91)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':42.0}, model_shortlist=80)
        ),
        'od-8930':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((800,1216), (800,1216), reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={}),
                model_path=f'../edgeai-modelzoo/models/vision/detection/coco/mmdet/fcos_r50-caffe_fpn_gn-head_ms-640-800-2x_coco.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, reshape_list=[(-1,5),(-1,1)], formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': None}, model_shortlist=100)
        ),
        'od-8940':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((800,1216), (800,1216), reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True,
                                                                        deny_list_from_start_end_node = {
                                                                            '/Sigmoid_1':None,
                                                                            '/Sigmoid_4':None,
                                                                            '/Sigmoid_3':None,
                                                                            '/Sigmoid':None,
                                                                            '/Sigmoid_2':None,
                                                                            '/bbox_head/Clip':None,
                                                                            '/bbox_head/Clip_2':None,
                                                                            '/bbox_head/Clip_3':None,}),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={
                    }),
                model_path=f'../edgeai-modelzoo/models/vision/detection/coco/mmdet/centernet-update_r50-caffe_fpn_ms-1x.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, reshape_list=[(-1,5),(-1,1)], formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': None}, model_shortlist=100)
        ),
        'od-8950':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((448, 672), (448, 672), reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=False, 
                                                                        deny_list_from_start_end_node = {'/bbox_head/Sigmoid':None}),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={
                    'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL
                    }),
                model_path=f'../edgeai-modelzoo/models/vision/detection/coco/mmdet/centernet_r18_crop512.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, reshape_list=[(-1,5),(-1,1)], formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': None}, model_shortlist=100)
        ),     
        ################################# MMDetection Models ###########################
        #DETR mmdetection
        'od-8960':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((800, 800), (800, 800), resize_with_pad=False, backend='cv2'),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={
                     'object_detection:meta_arch_type': 6, 
                     'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL
                     }),
                model_path=f'../edgeai-modelzoo/models/vision/detection/coco/edgeai-mmdet/detr_r50_8xb2-150e_20240722_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 39.9}, model_shortlist=None)
        ),
        #efficientDET-B0-Lite
        'od-8970':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((512,512), (512,512), reverse_channels=False, resize_with_pad=True, backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir, input_optimization=False, input_mean=(123.675, 116.28, 103.53), input_scale=(0.0171247538316637, 0.0175070028011204, 0.0174291938997821)),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={
                    'object_detection:meta_arch_type': 3,
                    'object_detection:meta_layers_names_list': f'../edgeai-modelzoo/models/vision/detection/coco/edgeai-mmdet/efficientdet_effb0_bifpn_lite_512x512_20240612_model.prototxt',
                    }),
                model_path=f'../edgeai-modelzoo/models/vision/detection/coco/edgeai-mmdet/efficientdet_effb0_bifpn_lite_512x512_20240612_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 32.2}, model_shortlist=None)
        ),
    }
    return pipeline_configs
