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
from . import descriptions


class ModelRunner():
    @classmethod
    def init_params(self, *args, **kwargs):
        params = init_params(*args, **kwargs)
        return params

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

        project_run_path_base = os.path.join(self.params.common.project_path, 'run')
        self.params.common.run_name = self.resolve_run_name(self.params.common.run_name)
        self.params.common.project_run_path = os.path.join(project_run_path_base, self.params.common.run_name, self.params.training.model_name)

        self.params.dataset.dataset_path = os.path.join(self.params.common.project_path, 'dataset')
        self.params.common.download_path = os.path.join(self.params.common.project_path, 'download')
        self.params.dataset.extract_path = self.params.dataset.dataset_path

        self.params.training.training_path = os.path.join(self.params.common.project_run_path, 'training')

        target_device_compilation_folder = self.params.common.target_device.lower()
        self.params.compilation.compilation_path = os.path.join(self.params.common.project_run_path, 'compilation', target_device_compilation_folder)

        if self.params.common.target_device in self.params.training.target_devices:
            target_device_data = self.params.training.target_devices[self.params.common.target_device]
            performance_fps = target_device_data['performance_fps']
            print(f'Model:{self.params.training.model_name} TargetDevice:{self.params.common.target_device} FPS(Estimate):{performance_fps}')
        #

    def resolve_run_name(self, run_name):
        if not run_name:
            return ''
        #
        # modify or set any parameters here as required.
        if '{date-time}' in run_name:
            run_name = run_name.replace('{date-time}', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        #
        return run_name

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
        
        # fetch the pretrained checkpoint if it is a url.
        # if it is a url and the downloaded copy is present, it will be reused.
        pretrained_path = self.params.training.pretrained_checkpoint_path
        if isinstance(pretrained_path, str) and (pretrained_path.startswith('http://') or pretrained_path.startswith('https://')):
            download_root = self.params.common.download_path
            download_success, exception_message, pretrained_path = utils.download_file(
                pretrained_path, download_root, extract=False)
            if download_success:
                self.params.training.pretrained_checkpoint_path = pretrained_path
            else:
                print('ERROR: Pretrained checkpoint could not be downloaded')
                return None
            #
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
            with open(self.params.training.log_file_path, 'a') as lfp:
                lfp.write('\nSUCCESS: ModelMaker - Training completed.')
            #
        #

        # actual model compilation
        if self.params.compilation.enable:
            self.model_compilation.clear()
            self.model_compilation.run()
            with open(self.params.compilation.log_file_path, 'a') as lfp:
                lfp.write('\nSUCCESS: ModelMaker - Compilation completed.')
            #
        #
        return self.params

    def get_params(self):
        return self.params

    @staticmethod
    def get_training_module_descriptions(*args, **kwargs):
        return descriptions.get_training_module_descriptions(*args, **kwargs)

    @staticmethod
    def get_model_descriptions(*args, **kwargs):
        return descriptions.get_model_descriptions(*args, **kwargs)

    @staticmethod
    def get_model_description(*args, **kwargs):
        return descriptions.get_model_description(*args, **kwargs)

    @staticmethod
    def set_model_description(*args, **kwargs):
        return descriptions.set_model_description(*args, **kwargs)

    @staticmethod
    def get_preset_descriptions(*args, **kwargs):
        return descriptions.get_preset_descriptions(*args, **kwargs)

    @staticmethod
    def get_target_device_descriptions(*args, **kwargs):
        return descriptions.get_target_device_descriptions(*args, **kwargs)

    @staticmethod
    def get_task_descriptions(*args, **kwargs):
        return descriptions.get_task_descriptions(*args, **kwargs)

    @staticmethod
    def get_sample_dataset_descriptions(*args, **kwargs):
        return descriptions.get_sample_dataset_descriptions(*args, **kwargs)
