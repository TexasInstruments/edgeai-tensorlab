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

import os

from edgeai_benchmark import utils
from edgeai_benchmark import datasets
from edgeai_benchmark.pipelines.pipeline_runner import PipelineRunner

from . import classification
from . import classification_v2
from . import detection
from . import detection_v2
from . import detection_additional
from . import face_detection
from . import segmentation
from . import human_pose_estimation
from . import object_6d_pose_estimation
from . import depth_estimation
from . import high_resolution
from . import miscllaneous
from . import detection_3d


def get_configs(settings, work_dir):
    # initialize the dataset place holders.
    if settings.dataset_cache is None:
        settings.dataset_cache = datasets.initialize_datasets(settings)
    #

    pipeline_configs = {}
    # merge all the config dictionaries
    pipeline_configs.update(classification.get_configs(settings, work_dir))
    pipeline_configs.update(classification_v2.get_configs(settings, work_dir))
    pipeline_configs.update(detection.get_configs(settings, work_dir))
    pipeline_configs.update(detection_v2.get_configs(settings, work_dir))
    pipeline_configs.update(face_detection.get_configs(settings, work_dir))
    pipeline_configs.update(segmentation.get_configs(settings, work_dir))
    pipeline_configs.update(human_pose_estimation.get_configs(settings,work_dir))
    pipeline_configs.update(object_6d_pose_estimation.get_configs(settings, work_dir))
    pipeline_configs.update(depth_estimation.get_configs(settings,work_dir))
    pipeline_configs.update(high_resolution.get_configs(settings,work_dir))
    pipeline_configs.update(miscllaneous.get_configs(settings,work_dir))
    pipeline_configs.update(detection_3d.get_configs(settings,work_dir))

    if settings.experimental_models:
        from . import detection_additional
        # now get the additional configs
        pipeline_configs.update(detection_additional.get_configs(settings, work_dir))
    #

    if settings.experimental_models:
        from . import classification_experimental
        from . import detection_experimental
        from . import face_detection_experimental
        from . import segmentation_experimental
        from . import human_pose_estimation_experimental
        from . import detection_3d_experimental
        from . import stereo_disparity_experimental
        # now get the experimental configs
        pipeline_configs.update(classification_experimental.get_configs(settings, work_dir))
        pipeline_configs.update(detection_experimental.get_configs(settings, work_dir))
        pipeline_configs.update(face_detection_experimental.get_configs(settings, work_dir))
        pipeline_configs.update(segmentation_experimental.get_configs(settings, work_dir))
        pipeline_configs.update(human_pose_estimation_experimental.get_configs(settings, work_dir))
        pipeline_configs.update(detection_3d_experimental.get_configs(settings, work_dir))
        pipeline_configs.update(stereo_disparity_experimental.get_configs(settings, work_dir))
    #

    if settings.external_models_path:
        external_models_configs = utils.import_file_folder(settings.external_models_path)
        pipeline_configs.update(external_models_configs.get_configs(settings, work_dir))
    #

    return pipeline_configs


def select_configs(settings, work_dir, session_name=None, remove_models=False):
    pipeline_configs = get_configs(settings, work_dir)
    pipeline_configs = PipelineRunner(settings, pipeline_configs).get_pipeline_configs()
    if session_name is not None:
        pipeline_configs = {pipeline_id:pipeline_config for pipeline_id, pipeline_config in pipeline_configs.items() \
                if pipeline_config['session'].peek_param('session_name') == session_name}
    #
    if remove_models:
        pipeline_configs = {pipeline_id:pipeline_config for pipeline_id, pipeline_config in pipeline_configs.items() \
                if os.path.exists(os.path.join(pipeline_config['session'].peek_param('run_dir'), 'param.yaml')) or
                   os.path.exists(pipeline_config['session'].peek_param('run_dir')+'.tar.gz') }
    #
    return pipeline_configs


def print_all_configs(pipeline_configs=None, enable_print=False):
    if enable_print:
        for k, v in sorted(pipeline_configs.items()):
            model_name = k + '-' + v['session'].kwargs['session_name']
            print("'{}',".format(model_name))
    return




