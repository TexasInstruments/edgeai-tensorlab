#################################################################################
# Copyright (c) 2018-2022, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################

import json
import os
import datetime

from . import constants
from ... import utils
from . import datasets
from . import training
from . import compilation
from .params import init_params


def get_model_descriptions(params):
    # populate a good pretrained model for the given task
    model_descriptions = training.get_model_descriptions(task_type=params.common.task_type,
                                                         target_device=params.common.target_device,
                                                         training_device=params.training.training_device)
    return model_descriptions


def get_model_description(params, model_key=None):
    model_key = (model_key or params.training.model_key)
    assert model_key, 'model_key must be specified for get_model_description().' \
                      'if model_key is not known, use the method get_model_descriptions() that returns all models.'
    model_description = training.get_model_description(model_key=model_key)
    return model_description


def set_model_description(params, model_description):
    assert model_description is not None, f'could not find pretrained model for {params.training.model_key}'
    assert params.common.task_type == model_description['common']['task_type'], \
        f'task_type: {params.common.task_type} does not match the pretrained model'
    # get pretrained model checkpoint and other details
    params.update(model_description)
    return params


class ModelRunner():
    @classmethod
    def init_params(self, *args, **kwargs):
        params = init_params(*args, **kwargs)
        return params

    @staticmethod
    def get_model_descriptions(*args, **kwargs):
        return get_model_descriptions(*args, **kwargs)

    @staticmethod
    def get_model_description(*args, **kwargs):
        return get_model_description(*args, **kwargs)

    @staticmethod
    def set_model_description(*args, **kwargs):
        return set_model_description(*args, **kwargs)

    def __init__(self, *args, verbose=True, **kwargs):
        self.params = self.init_params(*args, **kwargs)

        # print the runner params
        if verbose:
            [print(key, ':', value) for key, value in vars(self.params).items()]
        #
        # normalize the paths
        self.params.common.projects_path = utils.absolute_path(self.params.common.projects_path)
        self.params.dataset.input_data_path = utils.absolute_path(self.params.dataset.input_data_path)
        self.params.dataset.input_annotation_path = utils.absolute_path(self.params.dataset.input_annotation_path)
        self.params.common.project_path = os.path.join(self.params.common.projects_path, self.params.dataset.dataset_name)

        run_folder = self.params.common.run_name if self.params.common.run_name else ''
        self.params.common.project_run_path = os.path.join(self.params.common.project_path, 'run', run_folder)

        self.params.dataset.dataset_path = os.path.join(self.params.common.project_path, 'dataset')
        self.params.dataset.download_path = os.path.join(self.params.common.project_path, 'other', 'download')
        self.params.dataset.extract_path = self.params.dataset.dataset_path

        self.params.training.training_path = os.path.join(self.params.common.project_run_path, 'training', self.params.training.model_key)
        self.params.compilation.compilation_path = os.path.join(self.params.common.project_run_path, 'compilation')

        if self.params.common.target_device in self.params.training.target_devices:
            target_device_data = self.params.training.target_devices[self.params.common.target_device]
            performance_fps = target_device_data['performance_fps']
            print(f'Model:{self.params.training.model_key} TargetDevice:{self.params.common.target_device} FPS(Estimate):{performance_fps}')
        #

    def clear(self):
        pass

    def prepare(self):
        # create folders
        os.makedirs(self.params.common.project_path, exist_ok=True)
        os.makedirs(self.params.common.project_run_path, exist_ok=True)

        #####################################################################
        # prepare for dataset handling (loading, splitting, limiting files etc).
        self.dataset_handling = datasets.DatasetHandling(self.params)
        self.params.update(self.dataset_handling.get_params())
        # actual dataset handling
        if self.params.dataset.enable:
            self.dataset_handling.clear()
            self.dataset_handling.run()
        #
        
        # prepare model training
        self.training_target_module = training.get_target_module(self.params.training.training_backend,
                                                              self.params.common.task_type)
        self.model_training = self.training_target_module.ModelTraining(self.params)
        self.params.update(self.model_training.get_params())

        # prepare for model compilation
        self.model_compilation = compilation.edgeai_benchmark.ModelCompilation(self.params)
        self.params.update(self.model_compilation.get_params())

        # write out the description of the current run
        run_params_file = os.path.join(self.params.common.project_run_path, 'run.yaml')
        utils.write_dict(self.params, run_params_file)
        return run_params_file

    def run(self):
        # actual model training
        if self.params.training.enable:
            self.model_training.clear()
            self.model_training.run()
        #

        # actual model compilation
        if self.params.compilation.enable:
            self.model_compilation.clear()
            self.model_compilation.run()
        #
        return self.params

    def get_params(self):
        return self.params
