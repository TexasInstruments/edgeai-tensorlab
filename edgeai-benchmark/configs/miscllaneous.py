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

    robokitseg_cfg = {
        'task_type': 'visual_localization',
        'dataset_category': datasets.DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_TI_ROBOKIT_VISLOC_ZED1HD]['input_dataset'],
    }

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################mlperf models###################################
        ###############robokit visual localization model######################
        'visloc-7500': utils.dict_update(robokitseg_cfg,
            preprocess=preproc_transforms.get_transform_jai((384,768), (384,768), backend='cv2', interpolation=cv2.INTER_LINEAR),
            session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=settings.runtime_options_onnx_qat_v1(),
                model_path=f'{settings.models_path}/vision/visual_localization/carla/edgeai-tv/tiad_dkaze_carla_768x384_model.onnx'),
            postprocess=postproc_transforms.get_transform_none(),
            model_info=dict(metric_reference={'accuracy_localization%':None}, model_shortlist=10, compact_name='tiad-dkaze-carla-768x384', shortlisted=False)
        ),
    }
    return pipeline_configs

