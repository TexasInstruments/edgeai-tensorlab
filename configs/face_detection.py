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

    preproc_transforms = preprocess.PreProcessTransforms(settings)
    postproc_transforms = postprocess.PostProcessTransforms(settings)

    # for tflite models od post proc options can be specified in runtime_options
    # for onnx od models, od post proc options are specified in the prototxt
    # use a large top_k, keep_top_k and low confidence_threshold for accuracy measurement
    runtime_options_tflite_np2 = settings.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=False,
        runtime_options={'object_detection:confidence_threshold': settings.detection_thr,
                         'object_detection:nms_threshold': 0.45,
                         'object_detection:top_k': 200,
                         #'object_detection:keep_top_k': 100
                         })

    # configs for each model pipeline
    common_cfg = {
        'task_type': 'detection',
        'calibration_dataset': settings.dataset_cache['widerface']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['widerface']['input_dataset'],
    }

    common_session_cfg = sessions.get_common_session_cfg(settings, work_dir=work_dir)
    onnx_session_cfg = sessions.get_onnx_session_cfg(settings, work_dir=work_dir)
    onnx_bgr_session_cfg = sessions.get_onnx_bgr_session_cfg(settings, work_dir=work_dir)
    onnx_quant_session_cfg = sessions.get_onnx_quant_session_cfg(settings, work_dir=work_dir)
    onnx_bgr_quant_session_cfg = sessions.get_onnx_bgr_quant_session_cfg(settings, work_dir=work_dir)
    jai_session_cfg = sessions.get_jai_session_cfg(settings, work_dir=work_dir)
    jai_quant_session_cfg = sessions.get_jai_quant_session_cfg(settings, work_dir=work_dir)
    mxnet_session_cfg = sessions.get_mxnet_session_cfg(settings, work_dir=work_dir)
    tflite_session_cfg = sessions.get_tflite_session_cfg(settings, work_dir=work_dir)
    tflite_quant_session_cfg = sessions.get_tflite_quant_session_cfg(settings, work_dir=work_dir)

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
            preprocess=preproc_transforms.get_transform_onnx(416, 416, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(),
                                       {'object_detection:meta_arch_type': 6,
                                        'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/widerface/edgeai-mmdet/yolox_tiny_lite_416x416_20220318_model.prototxt',
                                        'advanced_options:output_feature_16bit_names_list': '1501, 1179, 1180, 1181, 1195, 1196, 1197, 1211, 1212, 1213'
                                        }),
                model_path=f'{settings.models_path}/vision/detection/widerface/edgeai-mmdet/yolox_tiny_lite_416x416_20220318_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.widerfacedet_det_label_offset_1to1(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 23.5}, model_name='OD-8410-yolox-tiny-lite-mmdet-widerface-416x416')
        ),
        'od-8420':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(),
                                       {'object_detection:meta_arch_type': 6,
                                        'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/widerface/edgeai-mmdet/yolox_s_lite_640x640_20220307_model.prototxt',
                                        'advanced_options:output_feature_16bit_names_list': '996, 711, 712, 713, 727, 728, 728, 743, 744, 745'
                                        }),
                model_path=f'{settings.models_path}/vision/detection/widerface/edgeai-mmdet/yolox_s_lite_640x640_20220307_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.widerfacedet_det_label_offset_1to1(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 31.62}, model_name='OD-8420-yolox-s-lite-mmdet-widerface-640x640')
        ),
        'od-8421':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(1024, 1024, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(),
                                       {'object_detection:meta_arch_type': 6,
                                        'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/widerface/edgeai-mmdet/yolox_s_lite_1024x1024_20220317_model.prototxt',
                                        'advanced_options:output_feature_16bit_names_list': '1033, 711, 712, 713, 727, 728, 728, 743, 744, 745'
                                        }),
                model_path=f'{settings.models_path}/vision/detection/widerface/edgeai-mmdet/yolox_s_lite_1024x1024_20220317_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.widerfacedet_det_label_offset_1to1(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': None}, model_name='OD-8421-yolox-s-lite-mmdet-widerface-1024x1024')
        ),
        'od-8430':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(),
                                       {'object_detection:meta_arch_type': 6,
                                        'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/widerface/edgeai-mmdet/yolox_m_lite_640x640_20220318_model.prototxt',
                                        #'advanced_options:output_feature_16bit_names_list': '996, 711, 712, 713, 727, 728, 728, 743, 744, 745'
                                        }),
                model_path=f'{settings.models_path}/vision/detection/widerface/edgeai-mmdet/yolox_m_lite_640x640_20220318_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.widerfacedet_det_label_offset_1to1(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 33.6}, model_name='OD-8430-yolox-m-lite-mmdet-widerface-640x640')
        ),
        'od-8450':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, resize_with_pad=True, backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**utils.dict_update(onnx_session_cfg, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(),
                                       {'object_detection:meta_arch_type': 6,
                                        'object_detection:meta_layers_names_list': f'../edgeai-yolov5/pretrained_models/models/detection/widerface/edgeai-yolov5/yolov5s6_640_ti_lite_metaarch.prototxt',
                                        'advanced_options:output_feature_16bit_names_list': '168, 370, 432, 494, 556'
                                        }),
                model_path=f'../edgeai-yolov5/pretrained_models/models/detection/widerface/edgeai-yolov5/yolov5s6_640_ti_lite_71p53.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.widerfacedet_det_label_offset_1to1(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 37.74})
        ),

    }
    return pipeline_configs