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

import cv2
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
    cityscapes_cfg = {
        'pipeline_type': settings.pipeline_type,
        'task_type': 'segmentation',
        'verbose': settings.verbose,
        'target_device': settings.target_device,
        'run_import': settings.run_import,
        'run_inference': settings.run_inference,
        'calibration_dataset': settings.dataset_cache['cityscapes']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['cityscapes']['input_dataset'],
    }

    ade20k_cfg = {
        'pipeline_type': settings.pipeline_type,
        'task_type': 'segmentation',
        'verbose': settings.verbose,
        'target_device': settings.target_device,
        'run_import': settings.run_import,
        'run_inference': settings.run_inference,
        'calibration_dataset': settings.dataset_cache['ade20k']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['ade20k']['input_dataset'],
    }

    ade20k_cfg_class32 = {
        'pipeline_type': settings.pipeline_type,
        'task_type': 'segmentation',
        'verbose': settings.verbose,
        'target_device': settings.target_device,
        'run_import': settings.run_import,
        'run_inference': settings.run_inference,
        'calibration_dataset': settings.dataset_cache['ade20k_class32']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['ade20k_class32']['input_dataset'],
    }

    pascal_voc_cfg = {
        'pipeline_type': settings.pipeline_type,
        'task_type': 'segmentation',
        'verbose': settings.verbose,
        'target_device': settings.target_device,
        'run_import': settings.run_import,
        'run_inference': settings.run_inference,
        'calibration_dataset': settings.dataset_cache['voc2012']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['voc2012']['input_dataset'],
    }

    common_session_cfg = dict(work_dir=work_dir, target_device=settings.target_device)

    postproc_segmentation_onnx = settings.get_postproc_segmentation_onnx()
    postproc_segmenation_tflite = settings.get_postproc_segmentation_tflite(with_argmax=False)

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################mlperf models###################################
        ################# jacinto-ai ONNX models : ADE20k-Class32 ###################################
        'vseg-18-100-0':utils.dict_update(ade20k_cfg_class32,
            preprocess=settings.get_preproc_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**common_session_cfg, **onnx_session_cfg,
                model_path=f'{settings.modelzoo_path}/vision/segmentation/ade20k_class32/jai-pytorch/deeplabv3lite_mobilenetv2_tv_512x512_ade20k_class32_20210308-092104.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':51.22})
        ),
        #################################################################
        #       TFLITE MODELS
        #################mlperf models###################################
        #mlperf: ade20k-segmentation (32 class) - deeplabv3_mnv2_ade20k_float - expected_metric??
        'vseg-18-010-0':utils.dict_update(ade20k_cfg_class32,
            preprocess=settings.get_preproc_tflite((512, 512), (512, 512), mean=(123.675, 116.28, 103.53), scale=(0.017125, 0.017507, 0.017429), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, **tflite_session_cfg,
                 model_path=f'{settings.modelzoo_path}/vision/segmentation/ade20k_class32/mlperf/deeplabv3_mnv2_ade20k_float.tflite'),
            postprocess=postproc_segmenation_tflite,
            model_info=dict(metric_reference={'accuracy_mean_iou%':54.8})
        ),
    }
    return pipeline_configs

