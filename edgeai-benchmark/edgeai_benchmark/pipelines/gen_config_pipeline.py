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
import yaml
import time
import itertools
from .. import utils, constants
from .base_pipeline import BasePipeline


class GenConfigPipeline(BasePipeline):
    def __init__(self, settings, pipeline_config):
        super().__init__(settings, pipeline_config)
        model_path = pipeline_config['session'].kwargs['model_path']
        model_path_wo_ext = os.path.splitext(model_path)[0]
        self.config_yaml = model_path_wo_ext + '_config.yaml'

    def __call__(self, description=''):
        model_id = self.pipeline_config['session'].kwargs['model_id']
        model_path = self.pipeline_config['session'].kwargs['model_path']
        model_proto_path = self.pipeline_config['session'].kwargs['runtime_options'].get('object_detection:meta_layers_names_list', None)
        write_gen_config = self.pipeline_config.get('write_gen_config', True)

        print(utils.log_color('\nINFO', 'running', f'{model_id}: {model_path}'))

        param_template = None
        if self.settings.param_template_file is not None:
            with open(self.settings.param_template_file) as fp:
                param_template = yaml.safe_load(fp)
            #
        #

        if model_path is None or not os.path.exists(model_path):
            return None

        self.pipeline_config['session'].kwargs['model_path'] = os.path.basename(model_path)
        self.pipeline_config['session'].kwargs['artifacts_folder'] = os.path.basename(self.pipeline_config['session'].kwargs['artifacts_folder'])
        if model_proto_path is not None:
            self.pipeline_config['session'].kwargs['runtime_options']['object_detection:meta_layers_names_list'] = os.path.basename(model_proto_path)
        #
        # config file is not specific to any device.
        # The one who uses the config file should spcify the device in his settings.
        self.pipeline_config['session'].kwargs['target_device'] = None
        pipeline_param = utils.pretty_object(self.pipeline_config)
        pipeline_param = utils.cleanup_dict(pipeline_param, param_template)
        if write_gen_config:
            with open(self.config_yaml, 'w') as fp:
                yaml.safe_dump(pipeline_param, fp, sort_keys=False)
            #
        else:
            print(utils.log_color('\nWARNING', 'skip writing config as it is already written',f'{model_id}: {model_path}'))

        result_dict = {'model_id':model_id, 'success': write_gen_config, 'config_path': self.config_yaml, 'pipeline_param':pipeline_param}
        print(utils.log_color('\n\nSUCCESS', 'gen config', f'{result_dict}\n'))

        return result_dict
