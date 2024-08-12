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
        #yolov4_scaled
        'od-8800':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640,640,resize_with_pad=True, backend='cv2'),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir ,  input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627) ),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={
                    'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list':f'../edgeai-modelzoo-cl/models/vision/detection/coco/scaled-yolov4-gplv3/scaled-yolov4-csp_lite_640x640_20240220_model.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'/module_list.143/Conv2d/Conv_output_0',
                    #  "deny_list:layer_type":"Squeeze,Transpose,Div,Unsqueeze,NonZero,Flatten,Reshape,Sigmoid,Greater,Pow,Expand,Range,Equal,Split,GatherND,NonMaxSuppression,Shape,Where,Gather,Slice,ScatterND",
                    #  "deny_list:layer_name":"/module_list.144/Sub,/module_list.159/Sub,/module_list.174/Sub,/Sub,/Sub_1,/module_list.144/Mul,/module_list.144/Mul_6,/module_list.144/Mul_7,/module_list.144/Mul_13,/module_list.159/Mul,/module_list.159/Mul_6,/module_list.159/Mul_7,/module_list.159/Mul_13,/module_list.174/Mul,/module_list.174/Mul_6,/module_list.174/Mul_7,/module_list.174/Mul_13,/Mul,/Mul_13,/Mul_14,/module_list.144/Add,/module_list.159/Add,/module_list.174/Add,/Add,/Add_1,/Add_4,/Add_6,/Add_8,/Add_9,/Add_3,/Add_5,/Add_7,/Add_10,/module_list.174/Mul,/module_list.174/Mul_6,/module_list.174/Mul_7,/module_list.174/Mul_13,/Mul,/Mul_13,/Mul_14,/module_list.174/Add,/Add,/Add_1,/Add_4,/Add_6,/Add_8,/Add_9,/Add_3,/Add_5,/Add_7,/Add_10,/Cast_8,/module_list.174/Sub,/Sub,/Sub_1,/Concat,/Concat_1,/Concat_3,/Concat_5,/Concat_7,/Concat_9,/Concat_11,/Concat_15"
                     }
                     ),
                model_path=f'../edgeai-modelzoo-cl/models/vision/detection/coco/scaled-yolov4-gplv3/scaled-yolov4-csp_lite_640x640_20240220_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True,formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':45.8}, model_shortlist=70)
        ),
        #yolov5-nano
         'od-8810':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640,  resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                     'object_detection:meta_layers_names_list':f'../edgeai-modelzoo-cl/models/vision/detection/coco/edgeai-mmyolo-gplv3/yolov5_nano-v61_lite_640x640_20240329_model.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'1,3,142,150,158'},
                     ),
                model_path=f'../edgeai-modelzoo-cl/models/vision/detection/coco/edgeai-mmyolo-gplv3/yolov5_nano-v61_lite_640x640_20240329_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 25.1}, model_shortlist=70)
        ),
        # yolov5-small
        'od-8820':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640,  resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                     'object_detection:meta_layers_names_list':f'../edgeai-modelzoo-cl/models/vision/detection/coco/edgeai-mmyolo-gplv3/yolov5_small-v61_lite_640x640_20240329_model.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'1,3,142,150,158',
                    #  "deny_list:layer_type":"Reshape,Slice,Transpose, Sigmoid, Gather, Unsqueeze, Sub, Pow, MatMul, NonMaxSuppression, Flatten, Shape, Expand, Equal, Tile, Where, ConstantOfShape, Neg, TopK", 
                    #  "deny_list:layer_name":"196,201,223,225,232,263,272,172,180,187,214,234,247,249,250,253,266,279,276,278,190,194,195,199,200,203,205,207,209,222,224,231,262,271",
                     },
                     fast_calibration=True),
                model_path=f'../edgeai-modelzoo-cl/models/vision/detection/coco/edgeai-mmyolo-gplv3/yolov5_small-v61_lite_640x640_20240329_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 35.5}, model_shortlist=70)
        ),
        #yolov6_n mmyolo
        # 'od-8840':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx(640, 640,  resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
        #     session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
        #         runtime_options=settings.runtime_options_onnx_np2(
        #             det_options=True, ext_options={'object_detection:meta_arch_type': 6,
        #              'object_detection:meta_layers_names_list':f'/data/files/a0508577/work/edgeai-algo/edgeai-mmyolo-gplv3/work_dirs/yolov6_n_syncbn_fast_8xb32-400e_coco/epoch_400.prototxt',
        #             #  'advanced_options:output_feature_16bit_names_list':'1,142,150,158'
        #              },
        #              fast_calibration=True),
        #         model_path=f'/data/files/a0508577/work/edgeai-algo/edgeai-mmyolo-gplv3/work_dirs/yolov6_n_syncbn_fast_8xb32-400e_coco/epoch_400.onnx'),
        #     postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 35.3}, model_shortlist=None)
        # ),
        #yolov7-tiny
        'od-8850':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640,  resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                     'object_detection:meta_layers_names_list':f'../edgeai-modelzoo-cl/models/vision/detection/coco/edgeai-mmyolo-gplv3/yolov7_tiny_lite_640x640_20230830_model.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'1,137,147,157',
                    #  "deny_list:layer_type": "MatMul,Gather,Cast,Transpose,ConstantOfShape,Sigmoid,Shape,Tile,NonMaxSuppression,Slice,Reshape,Expand,Equal,Pow,Where,Flatten,TopK,Unsqueeze,Neg,Sub",
                    #  "deny_list:layer_name": '/baseModel/head_module/convs_pred.2/convs_pred.2.2/Mul,/Mul,/Mul_3,/Mul_4,/Mul_5,/Mul_6,/Mul_7,/Mul_8,/Mul_9,/Mul_10,/Mul_11,/Mul_13,/Mul_14,/Mul_15,/Mul_16,/Concat_4,/Concat_5,/Concat_6,/Concat_7,/Concat_11,/Concat_12,/Concat_13,/Concat_14,/Concat_15,/Concat_17,/Concat_19,/Cast,/Cast_1,/Add_5,/Add_6,/Add_7,/Add_8,/Add_9,/Add_10,/Add_11'
                     },
                     fast_calibration=True),
                model_path=f'../edgeai-modelzoo-cl/models/vision/detection/coco/edgeai-mmyolo-gplv3/yolov7_tiny_lite_640x640_20230830_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':36.7}, model_shortlist=70)
        ),
        #yolov7-large
        'od-8860':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, resize_with_pad=True, backend='cv2',pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                     'object_detection:meta_layers_names_list':f'../edgeai-modelzoo-cl/models/vision/detection/coco/edgeai-mmyolo-gplv3/yolov7_large_lite_640x640_20240119_model.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'2,221,251,277',
                    #  "deny_list:layer_type":"Equal,Pow,TopK,Slice,ConstantOfShape,Unsqueeze,Flatten,Where,Transpose,Sigmoid,Reshape,Gather,Sub,MatMul,Expand,NonMaxSuppression,Shape,Neg,Tile",
                    #  "deny_list:layer_name":"/Cast,/Cast_1,/Concat,/Concat_1,/Concat_2,/Concat_3,/Concat_4,/Concat_5,/Concat_6,/Concat_7,/Concat_8,/Concat_9,/Concat_10,/Concat_11,/Concat_12,/Concat_13,/Concat_14,/Concat_15,/Concat_16,/Concat_17,/Concat_18,/Concat_19,/baseModel/head_module/convs_pred.2/convs_pred.2.2/Mul,/Mul,/Mul_1,/Mul_2,/Mul_3,/Mul_4,/Mul_5,/Mul_6,/Mul_7,/Mul_8,/Mul_9,/Mul_10,/Mul_11,/Mul_12,/Mul_13,/Mul_14,/Mul_15,/Mul_16,/baseModel/head_module/Constant_36,/baseModel/head_module/Constant_37,/baseModel/head_module/Constant_38,/baseModel/head_module/Constant_39,/baseModel/head_module/Constant_40,/baseModel/head_module/Constant_41,/baseModel/head_module/Constant_42,/baseModel/head_module/Constant_43,/baseModel/head_module/Constant_44,/baseModel/head_module/Constant_45,/baseModel/head_module/Constant_46,/baseModel/head_module/Constant_47,/baseModel/head_module/Constant_48,/baseModel/head_module/Constant_49,/baseModel/head_module/Constant_50,/baseModel/head_module/Constant_51,/Constant,/Constant_1,/Constant_2,/Constant_3,/Constant_4,/Constant_5,/Constant_6,/Constant_7,/Constant_8,/Constant_9,/Constant_10,/Constant_11,/Constant_12,/Constant_13,/Constant_14,/Constant_15,/Constant_16,/Constant_17,/Constant_18,/Constant_19,/Constant_20,/Constant_21,/Constant_22,/Constant_23,/Constant_24,/Constant_25,/Constant_26,/Constant_27,/Constant_28,/Constant_29,/Constant_30,/Constant_31,/Constant_32,/Constant_33,/Constant_34,/Constant_35,/Constant_36,/Constant_37,/Constant_38,/Constant_39,/Constant_40,/Constant_41,/Constant_42,/Constant_43,/Constant_44,/Constant_45,/Constant_46,/Constant_47,Constant_455,Constant_456,Constant_457,Constant_458,/Constant_48,/Constant_49,/Constant_50,/Constant_51,/Constant_52,/Constant_53,/Constant_54,/Constant_55,/Constant_56,/Constant_57,/Constant_58,/Constant_59,/Constant_60,/Constant_61,/Constant_62,/Constant_63,/Constant_64,/Constant_65,/Constant_66,/Constant_67,/Constant_68,/Constant_69,/Constant_70,/Constant_71,/Constant_72,/Add,/Add_1,/Add_2,/Add_3,/Add_4,/Add_5,/Add_6,/Add_7,/Add_8,/Add_9,/Add_10,/Add_11"
                     },
                     fast_calibration=True),
                model_path=f'../edgeai-modelzoo-cl/models/vision/detection/coco/edgeai-mmyolo-gplv3/yolov7_large_lite_640x640_20240119_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':48.1}, model_shortlist=70)
        ),
        #yolov8-nano
        'od-8870':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640,  resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                     det_options=True, ext_options={'object_detection:meta_arch_type': 8,
                     'object_detection:meta_layers_names_list':f'../edgeai-modelzoo-cl/models/vision/detection/coco/edgeai-mmyolo-gplv3/yolov8_nano_lite_640x640_20231118_model.prototxt',
                     'advanced_options:output_feature_16bit_names_list':''
                     },
                     ),
                model_path=f'../edgeai-modelzoo-cl/models/vision/detection/coco/edgeai-mmyolo-gplv3/yolov8_nano_lite_640x640_20231118_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':34.5}, model_shortlist=70)
        ),
        #yolov8-small
        'od-8880':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640,  resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                     det_options=True, ext_options={'object_detection:meta_arch_type': 8,
                     'object_detection:meta_layers_names_list':f'../edgeai-modelzoo-cl/models/vision/detection/coco/edgeai-mmyolo-gplv3/yolov8_small_lite_640x640_20231117_model.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'1,129,134,139,144,149,154',
                    #  "deny_list:layer_type":"Flatten,Sigmoid,Shape,MatMul,NonMaxSuppression,Sub,Neg,Squeeze,Where,Equal,Expand,ConstantOfShape,Transpose,Slice,TopK,Tile,Unsqueeze,Gather,Reshape,Softmax",
                    #  "deny_list:layer_name":"/Concat_4,/Concat_5,/Concat_6,/Concat_7,/Concat_8,/Concat_9,/Concat_13,/Concat_14,/Concat_15,/Concat_16,/Concat_17,/Concat_19,/Concat_21,/Cast_3,/Cast_4,/Mul,/Mul_1,/Mul_3,/Mul_4,/Mul_5,/Mul_6,/Add,/Add_1,/Add_2,/Add_3,/Add_4,/Add_5,/Add_6"
                     },
                     fast_calibration=True),
                model_path=f'../edgeai-modelzoo-cl/models/vision/detection/coco/edgeai-mmyolo-gplv3/yolov8_small_lite_640x640_20231117_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':42.4}, model_shortlist=70)
        ),
        # yolox_tiny lite versions from mmyolo
        'od-8890':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(416, 416, reverse_channels=True, resize_with_pad=[True,'corner'], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': f'../edgeai-modelzoo-cl/models/vision/detection/coco/edgeai-mmyolo-gplv3/yolox_tiny_lite_416x416_20231127_model.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '1,162,163,164,173,174,175,184,185,186',
                    # "deny_list:layer_type":"Neg,Expand,Shape,NonMaxSuppression,Slice,Reshape,Transpose,Exp,Sigmoid,ConstantOfShape,Flatten,Tile,MatMul,Where,Unsqueeze,Equal,TopK,Gather",
                    # "deny_list:layer_name":"/Add,/Add_1,/Add_2,/Add_3,/Add_4,/Add_5,/Mul,/Mul_1,/Mul_2,/Mul_3,/Mul_5,/Mul_6,/Mul_7,/Mul_8,/Cast_3,/Cast_4,/Concat_4,/Concat_5,/Concat_6,/Concat_7,/Concat_8,/Concat_9,/Concat_10,/Concat_14,/Concat_15,/Concat_16,/Concat_17,/Concat_18,/Concat_20,/Concat_22"
                    }),
                model_path=f'../edgeai-modelzoo-cl/models/vision/detection/coco/edgeai-mmyolo-gplv3/yolox_tiny_lite_416x416_20231127_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=[True,'corner'], formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 31.1}, model_shortlist=70)
        ),
        # yolox_small lite versions from mmyolo
        'od-8900':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=True, resize_with_pad=[True,'corner'], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': f'../edgeai-modelzoo-cl/models/vision/detection/coco/edgeai-mmyolo-gplv3/yolox_small_lite_640x640_20240319_model.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '1,162,163,164,173,174,175,184,185,186',
                    # "deny_list:layer_type":"TopK,Flatten,ConstantOfShape,Tile,MatMul,Unsqueeze,NonMaxSuppression,Neg,Where,Expand,Exp,Equal,Reshape,Shape,Slice,Sigmoid,Gather,Transpose",
                    # "deny_list:layer_name":"/Mul,/Mul_1,/Mul_2,/Mul_3,/Mul_5,/Mul_6,/Mul_7,/Mul_8,/Add,/Add_1,/Add_2,/Add_3,/Add_4,/Add_5,/Concat_4,/Concat_5,/Concat_6,/Concat_7,/Concat_8,/Concat_9,/Concat_10,/Concat_14,/Concat_15,/Concat_16,/Concat_17,/Concat_18,/Concat_20,/Concat_22,/Cast_3,/Cast_4"
                    }),
                model_path=f'../edgeai-modelzoo-cl/models/vision/detection/coco/edgeai-mmyolo-gplv3/yolox_small_lite_640x640_20240319_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=[True,'corner'], formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 38.7}, model_shortlist=70)
        ),
        #DETR_ResNet50
        'od-8910':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((800, 800),(800, 800), resize_with_pad=False, backend='cv2'),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, #input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)
                                                                      ),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                     'advanced_options:output_feature_16bit_names_list':'694, 698, 702',
                     'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL 
                     }),
                model_path=f'../edgeai-modelforest/models/vision/experimental/detr_resnet-50-simplified.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=False, reshape_list=None,logits_bbox_to_bbox_ls=True,formatter=postprocess.DetectionXYWH2XYXYCenterXY()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90(label_offset=0,num_classes=91)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':42.0}, model_shortlist=None)
        ),
        # Transformer models from huggingface transformers
        'od-8920':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((800,1066),(800,1066), resize_with_pad=True, backend='cv2'),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={
                     'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL 
                     }),
                model_path=f'{settings.models_path}/vision/detection/coco/hf-transformers/detr_resnet50_transformers_simp.onnx'),
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
            preprocess=preproc_transforms.get_transform_onnx((800, 800), (800, 800), resize_with_pad=False, backend='cv2', pad_color=[114, 114, 114]),
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
            preprocess=preproc_transforms.get_transform_onnx(512, 512, reverse_channels=False, resize_with_pad=True, backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir, input_optimization=False, input_mean=(123.675, 116.28, 103.53), input_scale=(0.0171247538316637, 0.0175070028011204, 0.0174291938997821)),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={
                    'object_detection:meta_arch_type': 3,
                    'object_detection:meta_layers_names_list': f'../edgeai-modelzoo/models/vision/detection/coco/edgeai-mmdet/efficientdet_effb0_bifpn_8xb16-crop512-300e_lite_20240612_model.prototxt',
                    }),
                model_path=f'../edgeai-modelzoo/models/vision/detection/coco/edgeai-mmdet/efficientdet_effb0_bifpn_8xb16-crop512-300e_lite_20240612_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 32.2}, model_shortlist=None)
        ),
    }
    return pipeline_configs
