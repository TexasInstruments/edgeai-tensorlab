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

from jai_benchmark import constants, utils, datasets, preprocess, sessions, postprocess, metrics


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
    common_cfg = {
        'task_type': '3d-detection',
        'calibration_dataset': settings.dataset_cache['kitti_lidar_det']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['kitti_lidar_det']['input_dataset'],
        'postprocess': None
    }

    common_session_cfg = dict(work_dir=work_dir, target_device=settings.target_device)

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        ################# onnx models ###############################
        # human pose estimation : mobilenetv2 + fpn_spp + udp, Expected AP : 42.31
        'lidar-3dod-7100':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_lidar_base(),
            session=onnx_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(),
                                    {'object_detection:meta_arch_type': 7,
                                     'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/3d_detection/kitti/mmdetection3d/lidar_point_pillars_496x432.prototxt',
                                     "advanced_options:add_data_convert_ops" : 0,
                                     }),
                model_path=f'{settings.models_path}/vision/3d_detection/kitti/mmdetection3d/lidar_point_pillars_496x432.onnx'),
            postprocess=postproc_transforms.get_transform_lidar_base(),
            metric=dict(label_offset_pred=None),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':47.1})
        )
    }
    return pipeline_configs
