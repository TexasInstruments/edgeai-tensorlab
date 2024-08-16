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


class BasePipeline():
    def __init__(self, settings, pipeline_config):
        self.info_dict = dict()
        self.settings = settings
        self.pipeline_config = pipeline_config
        self.avg_inference_time = None
        self.logger = None
        # run_dir is assigned after initialize is called in PipelineRunner
        # if it has not been created, it will be created in start
        self.session = self.pipeline_config['session']
        self.run_dir = self.session.get_param('run_dir')
        self.run_dir_base = os.path.split(self.run_dir)[-1]
        self.config_yaml = os.path.join(self.run_dir, 'config.yaml')
        # these files will be written after import and inference respectively
        self.param_yaml = os.path.join(self.run_dir, 'param.yaml')
        self.result_yaml = os.path.join(self.run_dir, 'result.yaml')
        # pop out dataset info from the pipeline config,
        # because it will increase the size of the para.yaml and result.yaml files
        if self.pipeline_config['input_dataset'] is not None:
            calibration_dataset = self.pipeline_config['calibration_dataset']
            if isinstance(calibration_dataset, dict):
                calibration_dataset.get_param('kwargs').pop('dataset_info', None)
            #
            self.dataset_info = None
            input_dataset = self.pipeline_config['input_dataset']
            if isinstance(input_dataset, dict):
                self.dataset_info = input_dataset.get_param('kwargs').pop('dataset_info', None)
            #
        else:
            self.dataset_info = None
        #
        if self.dataset_info is not None:
            self.dataset_info_file = os.path.join(self.run_dir, 'dataset.yaml')
            self.pipeline_config['input_dataset'].get_param('kwargs')['dataset_info'] = self.dataset_info_file
            self.pipeline_config['calibration_dataset'].get_param('kwargs')['dataset_info'] = self.dataset_info_file
        #

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        if self.logger is not None:
            self.logger.close()
            self.logger = None
        #

    def write_log(self, message):
        if self.logger is not None:
            self.logger.write(message)
        else:
            print(message)
