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
        # # edgeai-yolov5 models - IMPORTANT - see licence of the repository edgeai-yolov5 before using this model
        # 'od-8450expt':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx(640, 640, resize_with_pad=True, backend='cv2', pad_color=[114, 114, 114]),
        #     session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
        #         runtime_options=settings.runtime_options_onnx_np2(
        #             det_options=True, ext_options={'object_detection:meta_arch_type': 6,
        #             'object_detection:meta_layers_names_list': f'../edgeai-yolov5/pretrained_models/models/detection/widerface/edgeai-yolov5-gplv3/yolov5s6_640_ti_lite_metaarch.prototxt',
        #             'advanced_options:output_feature_16bit_names_list': '168, 370, 432, 494, 556'},
        #             fast_calibration=True),
        #         model_path=f'../edgeai-yolov5/pretrained_models/models/detection/widerface/edgeai-yolov5-gplv3/yolov5s6_640_ti_lite_71p53.onnx'),
        #     postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
        #     metric=dict(label_offset_pred=datasets.widerfacedet_det_label_offset_1to1(label_offset=1)),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 37.74}, model_shortlist=None)
        # ),
    }
    return pipeline_configs