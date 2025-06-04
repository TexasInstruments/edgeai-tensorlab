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


# for transformer models we need to set graph_optimization_level = ORT_DISABLE_ALL for onnxruntime
from onnxruntime import GraphOptimizationLevel
ORT_DISABLE_ALL = GraphOptimizationLevel.ORT_DISABLE_ALL


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
		# TODO: tidl_onnx_model_optimizer is switched off right now as it cannopt hanlde this particular variant of self attention
        'od-8920':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((800,800),(800,800), resize_with_pad=False, backend='cv2'),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=False),
                runtime_options=settings.runtime_options_onnx_np2(det_options=True, ext_options={
                    'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL,
                    'advanced_options:output_feature_16bit_names_list': '/bbox_predictor/layers.1/MatMul_output_0 onnx::MatMul_4038_netFormat /box_predictor/Relu_output_0 /box_predictor/Relu_1_output_0 /bbox_predictor/layers.2/Add_output_0 4053_netFormat 4041_netFormat'}),
                model_path=f'{settings.models_path}/vision/detection/coco/hf-transformers/detr_fb_resnet50_800x800_simp.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, reshape_list=[(-1,4),(-1,1),(-1,1)],logits_bbox_to_bbox_ls=True,formatter=postprocess.DetectionXYWH2XYXYCenterXY()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90(label_offset=0,num_classes=91)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':42.0}, model_shortlist=80, compact_name='DETR-fb-resnet50-transformer-coco-800x800', shortlisted=False)
        ),
        'od-8930':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((800,1216), (800,1216), reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=settings.runtime_options_onnx_np2(det_options=True, ext_options={}),
                model_path=f'../edgeai-modelzoo/models/vision/detection/coco/mmdet/fcos_r50-caffe_fpn_gn-head_ms-640-800-2x_coco.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, reshape_list=[(-1,5),(-1,1)], formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':38.7}, model_shortlist=100, compact_name='FCOS-r50-fpn-gn-head-coco-1216x800', shortlisted=False)
        ),
        'od-8940':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((800,1216), (800,1216), reverse_channels=True, resize_with_pad=False, backend='cv2'),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(103.53, 116.28, 123.675), input_scale=(1.0, 1.0, 1.0), input_optimization=False),
                runtime_options=settings.runtime_options_onnx_np2(det_options=True, ext_options={
                    'advanced_options:output_feature_16bit_names_list': '/bbox_head/conv_cls_2/Conv_output_0, /bbox_head/conv_cls_3/Conv_output_0, /bbox_head/conv_cls_4/Conv_output_0, /bbox_head/conv_cls/Conv_output_0, /bbox_head/conv_cls_1/Conv_output_0, /bbox_head/conv_reg_2/Conv_output_0, /bbox_head/conv_reg_3/Conv_output_0, /bbox_head/conv_reg_4/Conv_output_0, /bbox_head/conv_reg/Conv_output_0, /bbox_head/conv_reg_1/Conv_output_0',
                    'object_detection:meta_arch_type': 9,
                    'object_detection:meta_layers_names_list':f'../edgeai-modelzoo/models/vision/detection/coco/mmdet/centernet-update_r50-caffe_fpn_ms-1x.prototxt'}),
                model_path=f'../edgeai-modelzoo/models/vision/detection/coco/mmdet/centernet-update_r50-caffe_fpn_ms-1x.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, reshape_list=[(-1,5),(-1,1)], formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':40.2}, model_shortlist=100, compact_name='CenterNet-update-r50-fpn-coco-1216x800', shortlisted=False)
        ),
        'od-8950':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((448, 672), (448, 672), reverse_channels=False, resize_with_pad=False, backend='cv2'),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=settings.runtime_options_onnx_np2(det_options=True, ext_options={
                    'advanced_options:output_feature_16bit_names_list':'/bbox_head/heatmap_head/heatmap_head.2/Conv_output_0, /bbox_head/wh_head/wh_head.2/Conv_output_0, /bbox_head/offset_head/offset_head.2/Conv_output_0',
                    'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL,
                    'object_detection:meta_arch_type': 9,
                    'object_detection:meta_layers_names_list': f'../edgeai-modelzoo/models/vision/detection/coco/mmdet/centernet_r18_crop512.prototxt'}),
                model_path=f'../edgeai-modelzoo/models/vision/detection/coco/mmdet/centernet_r18_crop512.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, reshape_list=[(-1,5),(-1,1)], formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':25.9}, model_shortlist=100, compact_name='CenterNet-r18-coco-672x448', shortlisted=False)
        ),     
        ################################# MMDetection Models ###########################
        #efficientDET-B0-Lite
        'od-8970':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((512,512), (512,512), reverse_channels=False, resize_with_pad=True, backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir, input_optimization=False, input_mean=(123.675, 116.28, 103.53), input_scale=(0.0171247538316637, 0.0175070028011204, 0.0174291938997821)),
                runtime_options=settings.runtime_options_onnx_np2(det_options=True, ext_options={
                    'object_detection:meta_arch_type': 3,
                    'object_detection:meta_layers_names_list': f'../edgeai-modelzoo/models/vision/detection/coco/edgeai-mmdet/efficientdet_effb0_bifpn_lite_512x512_20240612_model.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '/bbox_head/reg_header/pointwise_conv/Conv_output_0, /bbox_head/cls_header/pointwise_conv/Conv_output_0, /bbox_head/reg_header/pointwise_conv_1/Conv_output_0 /bbox_head/cls_header/pointwise_conv_1/Conv_output_0 /bbox_head/reg_header/pointwise_conv_2/Conv_output_0 /bbox_head/cls_header/pointwise_conv_2/Conv_output_0 /bbox_head/reg_header/pointwise_conv_3/Conv_output_0 /bbox_head/cls_header/pointwise_conv_3/Conv_output_0 /bbox_head/reg_header/pointwise_conv_4/Conv_output_0 /bbox_head/cls_header/pointwise_conv_4/Conv_output_0', #/backbone/layers.0/activate/Relu_output_0
                    }),
                model_path=f'../edgeai-modelzoo/models/vision/detection/coco/edgeai-mmdet/efficientdet_effb0_bifpn_lite_512x512_20240612_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 32.2}, model_shortlist=100, compact_name='efficientDet-b0-bifpn-lite-coco-512x512', shortlisted=False)
        ),
        #yolov7_tiny_mmdet      
        # TODO : Add yolov7_tiny_lite and yolov7_tiny_original
        # 'od-9201':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx(640, 640, resize_with_pad=[True,'corner'], backend='cv2', pad_color=[114, 114, 114]),
        #     session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, #tidl_onnx_model_optimizer=True,
        #           input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627451, 0.003921568627451, 0.003921568627451)),
        #         runtime_options=settings.runtime_options_onnx_np2(det_options=True, ext_options={
        #           'object_detection:meta_arch_type': 6, 'object_detection:meta_layers_names_list':f''}),
        #         model_path=f''),
        #     postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=[True,'corner'], formatter=postprocess.DetectionBoxSL2BoxLS()),
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':30.7}, model_shortlist=100, compact_name='yolov7-l-mmdet-coco-640x640', shortlisted=False)
        # ),
        #yolov7_l_lite_mmdet
        'od-9202':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, resize_with_pad=[True,'corner'], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, #tidl_onnx_model_optimizer=True,
                    input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627451, 0.003921568627451, 0.003921568627451)),
                runtime_options=settings.runtime_options_onnx_np2(det_options=True, ext_options={
                    'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolov7_l_coco_lite_640x640_20250109_model.prototxt',
                    'advanced_options:output_feature_16bit_names_list':'231,232,228,229,225,226'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolov7_l_coco_lite_640x640_20250109_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=[True,'corner'], formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':45.9}, model_shortlist=70, compact_name='yolov7-l-lite-mmdet-coco-640x640', shortlisted=False)
        ),
        #yolov7_l_orig_mmdet
        'od-9203':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, resize_with_pad=[True,'corner'], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True,
                                                                      input_mean=(0.0, 0.0, 0.0),
                                                                      input_scale=(0.003921568627451, 0.003921568627451, 0.003921568627451)
                                                                        ),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                     'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolov7_l_coco_orig_640x640_20250109_model.prototxt',
                    #  'advanced_options:output_feature_16bit_names_list':''
                                                   }),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolov7_l_coco_orig_640x640_20250109_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=[True,'corner'], formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':50.3}, model_shortlist=70, compact_name='yolov7-l-mmdet-coco-640x640', shortlisted=False)
        ),
        #yolov9_s_lite_mmdet
        'od-9204':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=False, resize_with_pad=[True,'corner'], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False,
                                                                      input_mean=(0.0, 0.0, 0.0),
                                                                      input_scale=(0.003921568627451, 0.003921568627451, 0.003921568627451),
                                                                      deny_list_from_start_end_node = {
                                                                            '733':None,
                                                                            '753':None,
                                                                           }
                                                                          ),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    # 'object_detection:meta_layers_names_list':'',
                    'advanced_options:output_feature_16bit_names_list': ''
                    }),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolov9_s_coco_lite_640x640_20250219_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=[True,'corner'], formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 38.3}, model_shortlist=70, compact_name='yolov9-s-lite-mmdet-coco-640x640', shortlisted=True, recommended=True)
        ),
        #yolov9_s_plus_mmdet
        'od-9205':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=False, resize_with_pad=[True,'corner'], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True,
                                                                      input_mean=(0.0, 0.0, 0.0),
                                                                      input_scale=(0.003921568627451, 0.003921568627451, 0.003921568627451),
                                                                      deny_list_from_start_end_node = {
                                                                            '909':None,
                                                                            '929':None,
                                                                           }
                                                                          ),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={
                    'object_detection:meta_arch_type': 6,
                    #'object_detection:meta_layers_names_list':'',
                    'advanced_options:output_feature_16bit_names_list': '1,3'
                    }),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolov9_s_coco_plus_640x640_20250219_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=[True,'corner'], formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 40.0}, model_shortlist=70, compact_name='yolov9-s-plus-mmdet-coco-640x640', shortlisted=True, recommended=True)
        ),
        ######## rtmdet models  ########
        #rtmdet_m_lite
        'od-9206':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=True, resize_with_pad=[True,'corner'], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True,
                                                                      input_mean=(103.53, 116.28, 123.675),
                                                                      input_scale=(0.017429, 0.017507, 0.017125),
                                                                          ),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={
                    'object_detection:meta_arch_type': 8,
                    'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/rtmdet_m_coco_lite_640x640_20250404_model.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '1,3,254,267,280,259,272,285'
                    }),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/rtmdet_m_coco_lite_640x640_20250404_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=[True,'corner'], formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 40.0}, model_shortlist=None, compact_name='rtmdet-m-orig-mmdet-coco-640x640', shortlisted=True, recommended=True)
        ),
        #rtmdet_m_orig
        'od-9207':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=True, resize_with_pad=[True,'corner'], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True,
                                                                      input_mean=(103.53, 116.28, 123.675),
                                                                      input_scale=(0.017429, 0.017507, 0.017125),
                                                                          ),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={
                    'object_detection:meta_arch_type': 8,
                    'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/rtmdet_m_coco_orig_640x640_20250310_model.prototxt',
                    # 'advanced_options:output_feature_16bit_names_list': '1,4,40,97,154,199,353,370,387,360,377,394'
                    }),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/rtmdet_m_coco_orig_640x640_20250310_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=[True,'corner'], formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 40.0}, model_shortlist=None, compact_name='rtmdet-m-orig-mmdet-coco-640x640', shortlisted=True, recommended=True)
        ),
        #rtmdet_l_orig
        'od-9208':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=True, resize_with_pad=[True,'corner'], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True,
                                                                      input_mean=(103.53, 116.28, 123.675),
                                                                      input_scale=(0.017429, 0.017507, 0.017125),
                                                                          ),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={
                    'object_detection:meta_arch_type': 8,
                    'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/rtmdet_l_coco_orig_640x640_20250310_model.prototxt',
                    # 'advanced_options:output_feature_16bit_names_list': '1,3'
                    }),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/rtmdet_l_coco_orig_640x640_20250310_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=[True,'corner'], formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 40.0}, model_shortlist=None, compact_name='rtmdet-l-lite-mmdet-coco-640x640', shortlisted=True, recommended=True)
        ),
        #rtmdet_l_lite
        'od-9209':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=True, resize_with_pad=True, backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True,
                                                                      input_mean=(103.53, 116.28, 123.675),
                                                                      input_scale=(0.017429, 0.017507, 0.017125),
                                                                          ),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={
                    'object_detection:meta_arch_type': 8,
                    'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/rtmdet_l_coco_lite_640x640_20250310_model.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '1,3,187,319,332,345,324,337,350'
                    }),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/rtmdet_l_coco_lite_640x640_20250310_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 40.0}, model_shortlist=None, compact_name='rtmdet-l-lite-mmdet-coco-640x640', shortlisted=True, recommended=True)
        ),
    }
    return pipeline_configs
