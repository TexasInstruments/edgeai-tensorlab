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
        'dataset_category': datasets.DATASET_CATEGORY_WIDERFACE,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_WIDERFACE]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_WIDERFACE]['input_dataset'],
    }

    postproc_detection_onnx = postproc_transforms.get_transform_detection_onnx()
    postproc_detection_tflite = postproc_transforms.get_transform_detection_tflite()
    postproc_detection_efficientdet_ti_lite_tflite = postproc_transforms.get_transform_detection_tflite(normalized_detections=False, ignore_index=0,
                                                            formatter=postprocess.DetectionFormatting(dst_indices=(0,1,2,3,4,5), src_indices=(1,0,3,2,5,4)),
                                                            )
    postproc_detection_mxnet = postproc_transforms.get_transform_detection_mxnet()

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################onnx models#####################################
        'od-8410':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(416, 416, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/widerface/edgeai-mmdet/yolox_tiny_lite_416x416_20220318_model.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '1501, 1179, 1180, 1181, 1195, 1196, 1197, 1211, 1212, 1213'}),
                model_path=f'{settings.models_path}/vision/detection/widerface/edgeai-mmdet/yolox_tiny_lite_416x416_20220318_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.widerfacedet_det_label_offset_1to1(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 23.5, 'accuracy_ap50%': 47.4}, model_shortlist=20, compact_name='yolox-tiny-lite-mmdet-widerface-416x416', shortlisted=True, recommended=True)
        ),
        # for some reason, the model export has an issue in only 640x640 resolution - the exported model doesn't have NMS OP
        # this causes issue in SoCs that do not use tidl_offload
        'od-8420':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/widerface/edgeai-mmdet/yolox_s_lite_640x640_20220307_model.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '996, 711, 712, 713, 727, 728, 728, 743, 744, 745'}),
                model_path=f'{settings.models_path}/vision/detection/widerface/edgeai-mmdet/yolox_s_lite_640x640_20220307_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.widerfacedet_det_label_offset_1to1(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 31.62, 'accuracy_ap50%': 64.4}, model_shortlist=10, compact_name='yolox-s-lite-mmdet-widerface-640x640', shortlisted=True, recommended=True)
        ),
        'od-8421':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(1024, 1024, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/widerface/edgeai-mmdet/yolox_s_lite_1024x1024_20220317_model.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '1033, 711, 712, 713, 727, 728, 728, 743, 744, 745'}),
                model_path=f'{settings.models_path}/vision/detection/widerface/edgeai-mmdet/yolox_s_lite_1024x1024_20220317_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.widerfacedet_det_label_offset_1to1(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': None, 'accuracy_ap50%': 72.3}, model_shortlist=None, model_name='yolox-s-lite-mmdet-widerface-1024x1024')
        ),
        # more than model complexity, it seems input size helps in face detection - hence the above yolox_s 1024x1024 model is better than this
        # 'od-8430':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
        #     session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_np2(
        #             det_options=True, ext_options={'object_detection:meta_arch_type': 6,
        #             'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/widerface/edgeai-mmdet/yolox_m_lite_640x640_20220318_model.prototxt',
        #             #'advanced_options:output_feature_16bit_names_list': '996, 711, 712, 713, 727, 728, 728, 743, 744, 745'
        #             }),
        #         model_path=f'{settings.models_path}/vision/detection/widerface/edgeai-mmdet/yolox_m_lite_640x640_20220318_model.onnx'),
        #     postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
        #     metric=dict(label_offset_pred=datasets.widerfacedet_det_label_offset_1to1(label_offset=1)),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 33.6, 'accuracy_ap50%': 67.5}, model_shortlist=None, model_name='OD-8430-yolox-m-lite-mmdet-widerface-640x640')
        # ),
    }
    return pipeline_configs