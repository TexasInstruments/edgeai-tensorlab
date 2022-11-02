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
    # TIDL has post processing (simlar to object detection post processing) inside it for object 6d pose estimation
    # These models use that keypoint post processing
    # YOLO-6d-Pose: Enhancing YOLO for Multi Object 6D Pose Estimation
    # Debapriya Maji, Soyeb Nagori, Manu Mathew, Deepak Poddar
    # https://arxiv.org/abs/2204.06806
    common_cfg = {
        'task_type': 'object_6d_pose_estimation',
        'calibration_dataset': settings.dataset_cache['ycbv']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['ycbv']['input_dataset'],
        'postprocess': postproc_transforms.get_transform_detection_yolo_6d_object_pose_onnx()
    }

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        ################# onnx models ###############################
        # yolox based keypoint/pose estimation - post processing is handled completely by TIDL
        'op-7200':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
                        {'object_detection:meta_arch_type': 6,
                         'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/object_6d_pose/ycbv/edgeai-yolox/yolox_s_object_pose_ti_lite_metaarch.prototxt',
                        'advanced_options:output_feature_16bit_names_list': '597, 856, 1117, 1378',
                         "advanced_options:add_data_convert_ops": 0,
                        }),
                model_path=f'{settings.models_path}/vision/object_6d_pose/ycbv/edgeai-yolox/yolox_s_object_pose_ti_lite_64p15.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolo_6d_object_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), object6dpose=True),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':64.15})
        ),

        'op-7210':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx( (480,640), (480,640), reverse_channels=True, backend='cv2'),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
                        {'object_detection:meta_arch_type': 6,
                         'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/object_6d_pose/ycbv/edgeai-yolox/yolox_s_object_pose_ti_lite_metaarch_640x480.prototxt',
                        'advanced_options:output_feature_16bit_names_list': '597, 856, 1051, 1246',
                         "advanced_options:add_data_convert_ops": 0,
                        }),
                model_path=f'{settings.models_path}/vision/object_6d_pose/ycbv/edgeai-yolox/yolox_s_object_pose_ti_lite_640x480_64p15.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolo_6d_object_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), object6dpose=True),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':60.13})
        ),


    }
    return pipeline_configs