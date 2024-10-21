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
import sys
import copy
import argparse
import yaml
from .. import utils, preprocess, postprocess, pipelines, datasets, sessions, config_dict, config_settings, constants
from .run_accuracy import *

__all__ = ['run_configs_file']


def pipeline_param_to_config(settings, config_file_or_pipeline_param):
    if isinstance(config_file_or_pipeline_param, str):
        with open(config_file_or_pipeline_param) as cfp:
            pipeline_param = yaml.safe_load(cfp)
        #
        config_dirname = os.path.dirname(config_file_or_pipeline_param)
    else:
        pipeline_param = config_file_or_pipeline_param
        config_dirname = None
    #

    pipeline_config = copy.deepcopy(pipeline_param)
    session_name = pipeline_param['session']['session_name']
    task_type = pipeline_param['task_type']
    model_path = pipeline_param['session']['model_path']
    if model_path == os.path.basename(model_path) and config_dirname is not None:
        model_path = os.path.join(config_dirname, model_path)
        pipeline_param['session']['model_path'] = model_path
    #

    preprocess_pipeline = preprocess.PreProcessTransforms(settings).get_transform_base(**pipeline_param['preprocess'])
    pipeline_config['preprocess'] = preprocess_pipeline

    runtime_options = pipeline_param['session']['runtime_options']
    # override runtime_options if necessary
    if settings.target_device == constants.TARGET_DEVICE_TDA4VM:
        # handle unsupported case for TDA4VM
        if runtime_options['advanced_options:quantization_scale_type'] == constants.QUANT_SCALE_TYPE_NP2_PERCHAN:
            runtime_options['advanced_options:quantization_scale_type'] = constants.QUANT_SCALE_TYPE_P2
        #
    #
    if settings.calibration_frames is not None:
        runtime_options['advanced_options:calibration_frames'] = settings.calibration_frames
    #
    if settings.calibration_iterations is not None:
        runtime_options['advanced_options:calibration_iterations'] = settings.calibration_iterations
    #
    if settings.tensor_bits is not None:
        runtime_options['tensor_bits'] = settings.tensor_bits
    #
    pipeline_param['session']['runtime_options'] = runtime_options

    if session_name == constants.SESSION_NAME_ONNXRT:
        inference_session = sessions.ONNXRTSession(**pipeline_param['session'])
    elif session_name == constants.SESSION_NAME_TFLITERT:
        inference_session = sessions.ONNXRTSession(**pipeline_param['session'])
    elif session_name == constants.SESSION_NAME_TVMDLR:
        inference_session = sessions.TVMDLRSession(**pipeline_param['session'])
    #
    pipeline_config['session'] = inference_session

    if task_type == constants.TASK_TYPE_CLASSIFICATION:
        postprocess_pipeline = postprocess.PostProcessTransforms(settings).get_transform_classification(**pipeline_param['postprocess'])
    elif task_type == constants.TASK_TYPE_DETECTION:
        postprocess_pipeline = postprocess.PostProcessTransforms(settings).get_transform_detection_base(**pipeline_param['postprocess'])
    elif task_type == constants.TASK_TYPE_SEGMENTATION:
        postprocess_pipeline = postprocess.PostProcessTransforms(settings).get_transform_segmentation_base(**pipeline_param['postprocess'])
    elif task_type == constants.TASK_TYPE_KEYPOINT_DETECTION:
        postprocess_pipeline = postprocess.PostProcessTransforms(settings).get_transform_human_pose_estimation_base(**pipeline_param['postprocess'])
    elif task_type == constants.TASK_TYPE_DEPTH_ESTIMATION:
        postprocess_pipeline = postprocess.PostProcessTransforms(settings).get_transform_depth_estimation_base(**pipeline_param['postprocess'])
    elif task_type == constants.TASK_TYPE_DETECTION_3DOD:
        postprocess_pipeline = postprocess.PostProcessTransforms(settings).get_transform_detection_base(**pipeline_param['postprocess'])
    elif task_type == constants.TASK_TYPE_OBJECT_6D_POSE_ESTIMATION:
        postprocess_pipeline = postprocess.PostProcessTransforms(settings).get_transform_lidar_base(**pipeline_param['postprocess'])
    elif task_type == constants.TASK_TYPE_VISUAL_LOCALIZATION:
        postprocess_pipeline = postprocess.PostProcessTransforms(settings).get_transform_none(**pipeline_param['postprocess'])
    else:
        assert False, f'dont know how to construct pipeline config for task type: {task_type}, config: {config_file_or_pipeline_param}'
    #
    pipeline_config['postprocess'] = postprocess_pipeline

    return pipeline_config

def run_configs_file(settings, work_dir, pipeline_configs=None, modify_pipelines_func=None):
    # get the default configs if pipeline_configs is not given from outside
    if pipeline_configs is None:
        if settings.config_file is not None:
            configs_dict = {'configs': {'model-1': settings.config_file}}
        elif settings.configs_file is not None:
            configs_file = settings.configs_file
            if configs_file == os.path.basename(configs_file):
                configs_file = os.path.join(settings.models_path, configs_file)
            #
            with open(configs_file) as fp:
                configs_dict = yaml.safe_load(fp)
            #
        else:
            assert settings.configs_file is not None or settings.config_file, \
                f'either settings.configs_file or settings.config_file must be provided'
            configs_dict = {}
        #

        # read and create configs from configs_file
        pipeline_configs = {}
        for model_id, config_file in configs_dict['configs'].items():
            config_file_full_path = os.path.join(settings.models_path, config_file)
            pipeline_config = pipeline_param_to_config(settings, config_file_full_path)
            pipeline_configs[model_id] = pipeline_config
        #
    #

    results_list = run_accuracy(settings, work_dir, pipeline_configs, modify_pipelines_func)
    return results_list
