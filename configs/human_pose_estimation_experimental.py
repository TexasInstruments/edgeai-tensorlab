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

    # to define the names of first and last layer for 16 bit conversion
    first_last_layer = {
        'mobilenetv2_fpn_spp_udp': '363,561',
        'resnet50_fpn_spp_udp': '369,590',
        'mobilenetv2_pan_spp_udp': '669,1384', 
        'resnet50_pan_spp_udp': '675,1416' 
    }
    # get the sessions types to use for each model type
    onnx_session_type = settings.get_session_type(constants.MODEL_TYPE_ONNX)

    preproc_transforms = preprocess.PreProcessTransforms(settings)
    postproc_transforms = postprocess.PostProcessTransforms(settings)

    # configs for each model pipeline
    # these models need explicit post processing
    # this post process function used here implement post processing using associative embedding
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
        # yolov5 based keypoint/human pose estimation - post processing is handled completely by TIDL
        'kd-7040expt':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, resize_with_pad=True,  backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, input_mean=(0.0, 0.0, 0.0),  input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_p2(
                        det_options=True, ext_options={'tensor_bits': 16,
                         'object_detection:meta_arch_type': 6,
                         'object_detection:meta_layers_names_list': f'../edgeai-yolov5/pretrained_models/models/keypoint/coco/edgeai-yolov5-gplv3/yolov5s6_pose_640_ti_lite_metaarch.prototxt',
                         'advanced_options:calibration_iterations': 1}, fast_calibration=True),
                model_path=f'../edgeai-yolov5/pretrained_models/models/keypoint/coco/edgeai-yolov5-gplv3/yolov5s6_pose_640_ti_lite_54p9_82p2.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), keypoint=True),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':54.9}, model_shortlist=None)
        ),
        'kd-7050expt':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, input_mean=(0.0, 0.0, 0.0),  input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                        det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                         'object_detection:meta_layers_names_list': f'../edgeai-yolov5/pretrained_models/models/keypoint/coco/edgeai-yolov5-gplv3/yolov5s6_pose_640_ti_lite_metaarch.prototxt',
                         'advanced_options:output_feature_16bit_names_list': '176, 258,267, 335,333,328,326,349,347,342,340,363,361,356,354,377,375,370,368,  380,819,1258,1697'},
                         fast_calibration=True),
                model_path=f'../edgeai-yolov5/pretrained_models/models/keypoint/coco/edgeai-yolov5-gplv3/yolov5s6_pose_640_ti_lite_54p9_82p2.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_pose_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS(), keypoint=True),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':54.9}, model_shortlist=None)
        ),
        # edgeai-mmpose based models
        # human pose estimation : mobilenetv2 + fpn_spp + udp, Expected AP : 42.31
        'kd-7000expt':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=512, crop=512, resize_with_pad=True,
                backend='cv2', add_flip_image=settings.flip_test, pad_color=[127,127,127]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), {
                    'advanced_options:output_feature_16bit_names_list': first_last_layer['mobilenetv2_fpn_spp_udp']
                    }),
                model_path=f'{settings.models_path}/vision/keypoint/coco/edgeai-mmpose/mobilenetv2_fpn_spp_udp_512_20210610.onnx'),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':42.31})
        ),
        # human pose estimation : resnet50 + fpn_spp, Expected AP : 50.4
        'kd-7010expt':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=512, crop=512, resize_with_pad=True,
                backend='cv2', add_flip_image=settings.flip_test, pad_color=[127,127,127]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), {
                        'advanced_options:output_feature_16bit_names_list': first_last_layer['resnet50_fpn_spp_udp']
                        }),
                model_path=f'{settings.models_path}/vision/keypoint/coco/edgeai-mmpose/resnet50_fpn_spp_udp_512_20210610.onnx'),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':50.4})
        ),
        # human pose estimation : mobilenetv2 + pan_spp + udp, Expected AP : 45.41
        'kd-7020expt':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=512, crop=512, resize_with_pad=True, 
                backend='cv2', add_flip_image=settings.flip_test, pad_color=[127,127,127]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), {
                        'advanced_options:output_feature_16bit_names_list': first_last_layer['mobilenetv2_pan_spp_udp']
                        }),
                model_path=f'{settings.models_path}/vision/keypoint/coco/edgeai-mmpose/mobilenetv2_pan_spp_udp_512_20210617.onnx'),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':45.41})
        ),
        # human pose estimation : resnet50 + pan_spp + udp, Expected AP : 51.62
        'kd-7030expt':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=512, crop=512, resize_with_pad=True, 
                backend='cv2', add_flip_image=settings.flip_test, pad_color=[127,127,127]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(), {
                        'advanced_options:output_feature_16bit_names_list': first_last_layer['resnet50_pan_spp_udp']
                        }),
                model_path=f'{settings.models_path}/vision/keypoint/coco/edgeai-mmpose/resnet50_pan_spp_udp_512_20210616.onnx'),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':51.62})
        ),
    }
    return pipeline_configs
