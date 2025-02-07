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

from edgeai_benchmark import constants, utils, datasets, preprocess, sessions, postprocess, metrics


def get_configs(settings, work_dir):
    # get the sessions types to use for each model type
    onnx_session_type = settings.get_session_type(constants.MODEL_TYPE_ONNX)

    preproc_transforms = preprocess.PreProcessTransforms(settings)
    postproc_transforms = postprocess.PostProcessTransforms(settings)

    # configs for each model pipeline
    # TIDL has post processing (simlar to object detection post processing) inside it for keypoint estimation
    # These models use that keypoint post processing
    # YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss
    # Debapriya Maji, Soyeb Nagori, Manu Mathew, Deepak Poddar
    # https://arxiv.org/abs/2204.06806
    common_cfg = {
        'task_type': 'keypoint_detection',
        'dataset_category': datasets.DATASET_CATEGORY_COCOKPTS,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_COCOKPTS]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_COCOKPTS]['input_dataset'],
        'postprocess': postproc_transforms.get_transform_human_pose_estimation_onnx() 
    }

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        ################# onnx models ###############################
        # yolox based keypoint/pose estimation - post processing is handled completely by TIDL
        'kd-7060':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=settings.runtime_options_onnx_p2(
                        det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                         'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/keypoint/coco/edgeai-yolox/yolox_s_pose_ti_lite_640_20220301_model.prototxt',
                        'advanced_options:output_feature_16bit_names_list': '/0/backbone/backbone/stem/stem.0/act/Relu_output_0, /0/head/cls_preds.0/Conv_output_0, /0/head/reg_preds.0/Conv_output_0, /0/head/obj_preds.0/Conv_output_0, /0/head/kpts_preds.0/Conv_output_0, /0/head/cls_preds.1/Conv_output_0, /0/head/reg_preds.1/Conv_output_0, /0/head/obj_preds.1/Conv_output_0, /0/head/kpts_preds.1/Conv_output_0, /0/head/cls_preds.2/Conv_output_0, /0/head/reg_preds.2/Conv_output_0, /0/head/obj_preds.2/Conv_output_0, /0/head/kpts_preds.2/Conv_output_0'},
                        fast_calibration=True),
                model_path=f'{settings.models_path}/vision/keypoint/coco/edgeai-yolox/yolox_s_pose_ti_lite_640_20220301_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), keypoint=True),
            metric=dict(label_offset_pred=1), #TODO: add this for other models as well?
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':49.6, 'accuracy_ap50%':78.0}, model_shortlist=10, compact_name='human-pose-yolox-s-640x640', shortlisted=True, recommended=True)
        ),
        'kd-7070':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(416, 416, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir, input_optimization=False,
                                                                        deny_list_from_start_end_node = {
                                                                            '201':None,
                                                                            '224':None,
                                                                            '177':None,
                                                                            }
                                                                            ),
                runtime_options=settings.runtime_options_onnx_p2(
                        det_options=True, ext_options={
                        'object_detection:meta_arch_type': 6,
                        #  'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/keypoint/coco/edgeai-mmpose/yoloxpose_tiny_lite_416x416_20240808_model.prototxt',
                        'advanced_options:output_feature_16bit_names_list': '3'
                        },
                        fast_calibration=True),
                model_path=f'{settings.models_path}/vision/keypoint/coco/edgeai-mmpose/yoloxpose_tiny_lite_416x416_20240808_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), keypoint=True),
            metric=dict(label_offset_pred=1), #TODO: add this for other models as well?
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':47.2, 'accuracy_ap50%':76.0}, model_shortlist=100)
        ),
        'kd-7080':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir, input_optimization=False,
                                                                        deny_list_from_start_end_node = {
                                                                            '201':None,
                                                                            '224':None,
                                                                            '177':None,
                                                                            }
                                                                            ),
                runtime_options=settings.runtime_options_onnx_p2(
                        det_options=True, ext_options={
                        'object_detection:meta_arch_type': 6,
                        #  'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/keypoint/coco/edgeai-mmpose/yoloxpose_s_lite_coco-640x640_20250119_model.prototxt',
                        'advanced_options:output_feature_16bit_names_list': '3'
                        },
                        fast_calibration=True),
                model_path=f'{settings.models_path}/vision/keypoint/coco/edgeai-mmpose/yoloxpose_s_lite_coco-640x640_20250119_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), keypoint=True),
            metric=dict(label_offset_pred=1), #TODO: add this for other models as well?
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':56.4, 'accuracy_ap50%':83.4}, model_shortlist=100)
        ),
    }
    return pipeline_configs