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
from edgeai_benchmark import constants, utils, datasets, preprocess, sessions, postprocess, metrics


def get_configs(settings, work_dir):
    # get the sessions types to use for each model type
    onnx_session_type = settings.get_session_type(constants.MODEL_TYPE_ONNX)
    tflite_session_type = settings.get_session_type(constants.MODEL_TYPE_TFLITE)
    mxnet_session_type = settings.get_session_type(constants.MODEL_TYPE_MXNET)

    preproc_transforms = preprocess.PreProcessTransforms(settings)
    postproc_transforms = postprocess.PostProcessTransforms(settings)

    nyudepthv2_cfg = {
        'task_type': 'depth_estimation',
        'dataset_category': datasets.DATASET_CATEGORY_NYUDEPTHV2,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_NYUDEPTHV2]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_NYUDEPTHV2]['input_dataset'],
    }

    postproc_depth_estimation_onnx = postproc_transforms.get_transform_depth_estimation_onnx()

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################mlperf models###################################
        # edgeai: segmentation - fpnlite_aspp_regnetx400mf_ade20k32_384x384_20210314-205347 expected_metric: 51.03% mean-iou
        'de-7300':utils.dict_update(nyudepthv2_cfg,
            preprocess=preproc_transforms.get_transform_jai((246,246), (224,224), backend='cv2', interpolation=cv2.INTER_NEAREST),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0, 0, 0), input_scale=(1/255, 1/255, 1/255), input_optimization=False),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
                    {'advanced_options:output_feature_16bit_names_list':'233, 424'}),
                model_path=f'{settings.models_path}/vision/depth_estimation/nyudepthv2/fast-depth/fast-depth.onnx'),
            postprocess=postproc_depth_estimation_onnx,
            metric=dict(disparity=False, scale_shift=False),
            model_info=dict(metric_reference={'accuracy_delta_1%':77.1}, model_shortlist=50)
        ),
        'de-7310':utils.dict_update(nyudepthv2_cfg,
            preprocess=preproc_transforms.get_transform_jai((256,256), (256,256), backend='cv2', interpolation=cv2.INTER_CUBIC),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(123.675, 116.28, 103.53), input_scale=(0.017125, 0.017507, 0.017429), input_optimization=False),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
                    {'advanced_options:output_feature_16bit_names_list':'511, 983'}),
                model_path=f'{settings.models_path}/vision/depth_estimation/nyudepthv2/MiDaS/midas-small.onnx'),
            postprocess=postproc_depth_estimation_onnx,
            metric=dict(disparity=True, scale_shift=True),
            model_info=dict(metric_reference={'accuracy_delta_1%':86.4}, model_shortlist=50)
        ),
    }
    return pipeline_configs

