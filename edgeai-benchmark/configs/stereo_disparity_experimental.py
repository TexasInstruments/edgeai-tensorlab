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

    kitti2015_cfg = {
        'task_type': 'stereo_disparity',
        'dataset_category': datasets.DATASET_CATEGORY_KITTI_2015,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_KITTI_2015]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_KITTI_2015]['input_dataset'],
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

    postproc_stereo_disparity_onnx = postproc_transforms.get_transform_disparity_estimation_onnx()

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################mlperf models###################################        
        'sd-7400':utils.dict_update(kitti2015_cfg,
            preprocess=preproc_transforms.get_transform_jai(None, (368,1184), backend='cv2', interpolation=cv2.INTER_CUBIC),
            session=onnx_session_type(**utils.dict_update(onnx_session_cfg, input_mean=(123.93, 116.28, 103.53), input_scale=(0.017125, 0.017507, 0.017429)),
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(),
                    {'advanced_options:output_feature_16bit_names_list':'511, 983'}),
                model_path=f'{settings.models_path}/vision/stereo_disparity/kitti2015/gwcNet/gwcNet-gc.onnx'),
            postprocess=postproc_stereo_disparity_onnx,
            # To REVISIT!
            metric=dict(disparity=True),
            model_info=dict(metric_reference={'accuracy_pixel_error':0.7}, model_shortlist=None, compact_name='gwcNet-gc-stereo-kitti2015-1184x384', shortlisted=False)
        ),
    }
    return pipeline_configs

