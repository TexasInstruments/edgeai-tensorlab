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
from jacinto_ai_benchmark import constants, utils, datasets, preprocess, sessions, postprocess, metrics


def get_configs(settings, work_dir):
    # get the sessions types to use for each model type
    onnx_session_type = settings.get_session_type(constants.MODEL_TYPE_ONNX)
    tflite_session_type = settings.get_session_type(constants.MODEL_TYPE_TFLITE)
    mxnet_session_type = settings.get_session_type(constants.MODEL_TYPE_MXNET)

    # get the session cfgs to be used for float models
    onnx_session_cfg = settings.get_session_cfg(constants.MODEL_TYPE_ONNX, is_qat=False)
    tflite_session_cfg = settings.get_session_cfg(constants.MODEL_TYPE_TFLITE, is_qat=False)
    mxnet_session_cfg = settings.get_session_cfg(constants.MODEL_TYPE_MXNET, is_qat=False)

    # get the session cfgs to be used for qat models
    onnx_session_cfg_qat = settings.get_session_cfg(constants.MODEL_TYPE_ONNX, is_qat=True)
    tflite_session_cfg_qat = settings.get_session_cfg(constants.MODEL_TYPE_TFLITE, is_qat=True)
    mxnet_session_cfg_qat = settings.get_session_cfg(constants.MODEL_TYPE_MXNET, is_qat=True)

    # configs for each model pipeline
    common_cfg = {
        'pipeline_type': settings.pipeline_type,
        'task_type': 'detection',
        'verbose': settings.verbose,
        'target_device': settings.target_device,
        'run_import': settings.run_import,
        'run_inference': settings.run_inference,
        'calibration_dataset': settings.dataset_cache['coco']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['coco']['input_dataset'],
    }

    common_session_cfg = dict(work_dir=work_dir, target_device=settings.target_device)

    postproc_detection_onnx = settings.get_postproc_detection_onnx()
    postproc_detection_tflite = settings.get_postproc_detection_tflite()
    postproc_detection_mxnet = settings.get_postproc_detection_mxnet()

    pipeline_configs = {
        #################################################################
        #       MXNET MODELS
        #################################################################
        # mxnet : gluoncv model : detection - ssd_512_mobilenet1.0_coco - accuracy: 21.7% ap[0.5:0.95], 39.2% ap50
        'vdet-12-061-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((512,512), (512,512), backend='cv2'),
            session=mxnet_session_type(**common_session_cfg, **mxnet_session_cfg,
                model_path=[f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_mobilenet1.0_coco-symbol.json',
                            f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_mobilenet1.0_coco-0000.params'],
                model_type='mxnet', input_shape={'data':(1,3,512,512)}),
            postprocess=postproc_detection_mxnet,
            metric=dict(label_offset_pred=coco_label_offset_80to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':21.7})
        ),
        #################################################################
        #       TFLITE MODELS
        #################tflite models###################################
        # mlperf edge: detection - ssd_mobilenet_v1_coco_2018_01_28 expected_metric: 23.0% ap[0.5:0.95] accuracy
        'vdet-12-010-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((300,300), (300,300), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, **tflite_session_cfg,
                model_path=f'{settings.models_path}/vision/detection/coco/mlperf/ssd_mobilenet_v1_coco_2018_01_28.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=coco_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':23.0})
        ),
    }
    return pipeline_configs



################################################################################################
# convert from 80 class index (typical output of a mmdetection detector) to 90 or 91 class
# (original labels of coco starts from 1, and 0 is background)
# the evalation/metric script will convert from 80 class to coco's 90 class.
def coco_label_offset_80to90(label_offset=1):
    coco_label_table = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                         21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                         41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                         61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                         81, 82, 84, 85, 86, 87, 88, 89, 90]

    if label_offset == 1:
        # 0 => 1, 1 => 2, .. 79 -> 90, 80 => 91
        coco_label_offset = {k:v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({80:91})
    elif label_offset == 0:
        # 0 => 0, 1 => 1, .. 80 => 90
        coco_label_offset = {(k+1):v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({0:0})
    else:
        assert False, f'unsupported value for label_offset {label_offset}'
    #
    return coco_label_offset


# convert from 90 class index (typical output of a tensorflow detector) to 90 or 91 class
# (original labels of coco starts from 1, and 0 is background)
def coco_label_offset_90to90(label_offset=1):
    coco_label_table = range(1,91)
    if label_offset == 1:
        # 0 => 1, 1 => 2, .. 90 => 91
        coco_label_offset = {k:v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({-1:0,90:91})
    elif label_offset == 0:
        # 0 => 0, 1 => 1, .. 90 => 90
        coco_label_offset = {(k+1):v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({-1:-1,0:0})
    else:
        assert False, f'unsupported value for label_offset {label_offset}'
    #
    return coco_label_offset