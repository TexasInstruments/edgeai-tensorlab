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

from ..runtimes import config_dict
from .. import utils, preprocess, postprocess, pipelines, datasets, sessions, config_settings, constants

__all__ = ['get_configs_from_file']


def pipeline_param_to_config(settings, config_file_or_pipeline_param, work_dir, adjust_config):
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

    calibration_dataset = pipeline_config['calibration_dataset']
    if isinstance(calibration_dataset, dict):
        calibration_dataset_name = calibration_dataset.pop('type')
        calibration_dataset_type = getattr(datasets, calibration_dataset_name)
        calibration_dataset = calibration_dataset_type(**calibration_dataset)
    #
    pipeline_config['calibration_dataset'] = calibration_dataset

    input_dataset = pipeline_config['input_dataset']
    if isinstance(input_dataset, dict):
        input_dataset_name = input_dataset.pop('type')
        input_dataset_type = getattr(datasets, input_dataset_name)
        input_dataset = input_dataset_type(**input_dataset)
    #
    pipeline_config['input_dataset'] = input_dataset

    preprocess_pipeline = preprocess.PreProcessTransforms(settings).get_transform_base(**pipeline_param['preprocess'])
    pipeline_config['preprocess'] = preprocess_pipeline

    if adjust_config:
        if config_dirname is not None:
            model_path = pipeline_param['session']['model_path']
            if model_path == os.path.basename(model_path):
                model_path = os.path.join(config_dirname, model_path)
                pipeline_param['session']['model_path'] = model_path
            #

            meta_layers_names_list = pipeline_param['session']['runtime_options'].get('object_detection:meta_layers_names_list', None)
            if meta_layers_names_list is not None and meta_layers_names_list == os.path.basename(meta_layers_names_list):
                meta_layers_names_list = os.path.join(config_dirname, meta_layers_names_list)
                pipeline_param['session']['runtime_options']['object_detection:meta_layers_names_list'] = meta_layers_names_list
            #
        #

        runtime_options_in_config = pipeline_param['session']['runtime_options']
        runtime_options_in_settings = settings.get_runtime_options()
        runtime_options = copy.deepcopy(runtime_options_in_settings)
        # conditional update
        for rt_opt_key, rt_opt_value in runtime_options_in_config.items():
            if rt_opt_key in ['advanced_options:calibration_frames', 'advanced_options:calibration_iterations'] \
                and rt_opt_key in runtime_options_in_config:
                rt_opt_value = min(runtime_options_in_settings[rt_opt_key], runtime_options_in_config[rt_opt_key])
            #
            runtime_options[rt_opt_key] = rt_opt_value
        #

        # handle device specific overrides
        if settings.runtime_options is not None:
            runtime_options.update(settings.runtime_options)
        #
        if settings.target_device == constants.TARGET_DEVICE_TDA4VM:
            if runtime_options['advanced_options:quantization_scale_type'] == constants.QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN:
                runtime_options['advanced_options:quantization_scale_type'] = constants.QUANTScaleType.QUANT_SCALE_TYPE_P2
            #
        #
    #

    # set it to pipeline_params
    pipeline_param['session']['work_dir'] = work_dir
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

def get_configs_from_file(settings, work_dir, adjust_config):
    config_dict_or_file = settings.configs_path
    if isinstance(config_dict_or_file, str) and (os.path.splitext(config_dict_or_file)[-1] == '.yaml'):
        if config_dict_or_file == os.path.basename(config_dict_or_file):
            config_dict_or_file = os.path.abspath(os.path.join(settings.models_path, config_dict_or_file))
        #
        with open(config_dict_or_file) as fp:
            configs_dict = yaml.safe_load(fp)
            assert isinstance(configs_dict, dict), f'config file contnet must be a dict {config_file}'
        #
    elif isinstance(config_dict_or_file, dict):
        configs_dict = config_dict_or_file
    else:
        assert False, f"settings.configs_path must be a yaml file path or a dict: got {type(config_dict_or_file)}"
    #
    if 'configs' not in configs_dict:
        if 'session' in configs_dict:
            configs_dict = {'configs': {'model-1': configs_dict}}
        else:
            configs_dict = {'configs': configs_dict}
        #
    #

    # read and create configs from configs_file
    pipeline_configs = {}
    for model_id, config_entry in configs_dict['configs'].items():
        if isinstance(config_entry, str):
            config_entry = os.path.normpath(config_entry)
            if not config_entry.startswith(os.sep):
                config_entry = os.path.abspath(os.path.join(settings.models_path, config_entry))
            #
        else:
            assert isinstance(config_entry, dict), f'configs file entries must be files names if dict. got: {config_entry}'
        #
        pipeline_config = pipeline_param_to_config(settings, config_entry, work_dir, adjust_config=adjust_config)
        pipeline_configs[model_id] = pipeline_config
    #
    return pipeline_configs

def select_configs_from_file(settings, work_dir, session_name=None, remove_models=False, adjust_config=True):
    pipeline_configs = get_configs_from_file(settings, work_dir, adjust_config=adjust_config)
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
