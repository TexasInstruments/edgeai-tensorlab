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
        'task_type': 'human_pose_estimation',
        'dataset_category': datasets.DATASET_CATEGORY_COCOKPTS,
        'calibration_dataset': settings.dataset_cache['cocokpts']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['cocokpts']['input_dataset'],
        'postprocess': postproc_transforms.get_transform_human_pose_estimation_onnx() 
    }

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        ################# onnx models ###############################
        # yolov5 based keypoint/pose estimation - post processing is handled completely by TIDL
        'kd-7040':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, resize_with_pad=True,  backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, input_mean=(0.0, 0.0, 0.0),  input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_p2(
                        det_options=True, ext_options={'tensor_bits': 16,
                         'object_detection:meta_arch_type': 6,
                         'object_detection:meta_layers_names_list': f'../edgeai-yolov5/pretrained_models/models/keypoint/coco/edgeai-yolov5/yolov5s6_pose_640_ti_lite_metaarch.prototxt',
                         'advanced_options:calibration_iterations': 1}),
                model_path=f'../edgeai-yolov5/pretrained_models/models/keypoint/coco/edgeai-yolov5/yolov5s6_pose_640_ti_lite_54p9_82p2.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), keypoint=True),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':54.9}, model_shortlist=None)
        ),
        'kd-7050':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, input_mean=(0.0, 0.0, 0.0),  input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                        det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                         'object_detection:meta_layers_names_list': f'../edgeai-yolov5/pretrained_models/models/keypoint/coco/edgeai-yolov5/yolov5s6_pose_640_ti_lite_metaarch.prototxt',
                         'advanced_options:output_feature_16bit_names_list': '176, 258,267, 335,333,328,326,349,347,342,340,363,361,356,354,377,375,370,368,  380,819,1258,1697'}),
                model_path=f'../edgeai-yolov5/pretrained_models/models/keypoint/coco/edgeai-yolov5/yolov5s6_pose_640_ti_lite_54p9_82p2.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), keypoint=True),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':54.9}, model_shortlist=None)
        ),
        # yolox based keypoint/pose estimation - post processing is handled completely by TIDL
        'kd-7060':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=settings.runtime_options_onnx_p2(
                        det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                         'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/keypoint/coco/edgeai-yolox/yolox_s_pose_ti_lite_640_20220301_model.prototxt',
                        'advanced_options:output_feature_16bit_names_list': '513, 758, 883, 1008, 756, 753, 878, 881, 1003, 1006'}),
                model_path=f'{settings.models_path}/vision/keypoint/coco/edgeai-yolox/yolox_s_pose_ti_lite_640_20220301_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), keypoint=True),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':49.6, 'accuracy_ap50%':78.0}, model_shortlist=10)
        ),

    }
    return pipeline_configs