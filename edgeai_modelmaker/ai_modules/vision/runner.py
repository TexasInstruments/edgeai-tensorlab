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
import tarfile
import torch
import warnings
import copy
import yaml

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
        # set the checkpoint download folder
        # (for the models that are downloaded using torch.hub eg. mmdetection uses that)
        torch.hub.set_dir(os.path.join(params.common.download_path, 'pretrained', 'torch', 'hub'))
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

        self.params.common.run_name = self.resolve_run_name(self.params.common.run_name, self.params.training.model_name)
        self.params.common.project_run_path = os.path.join(self.params.common.project_path, 'run', self.params.common.run_name)

        self.params.dataset.dataset_path = os.path.join(self.params.common.project_path, 'dataset')
        self.params.dataset.extract_path = self.params.dataset.dataset_path

        self.params.training.training_path = os.path.join(self.params.common.project_run_path, 'training')
        self.params.training.model_packaged_path = os.path.join(self.params.training.training_path,
                                    '_'.join(os.path.split(self.params.common.run_name))+'.tar.gz')

        assert self.params.common.target_device in constants.TARGET_DEVICES_ALL, f'common.target_device must be set to one of: {constants.TARGET_DEVICES_ALL}'
        target_device_compilation_folder = self.params.common.target_device
        self.params.compilation.compilation_path = os.path.join(self.params.common.project_run_path, 'compilation', target_device_compilation_folder)

        if self.params.common.target_device in self.params.training.target_devices:
            performance_infer_time_ms_list = {k:v['performance_infer_time_ms'] for k,v in self.params.training.target_devices.items()}
            print('---------------------------------------------------------------------')
            print(f'INFO: ModelMaker - '
                    f'task_type:{self.params.common.task_type} '
                    f'model_name:{self.params.training.model_name} '
                    f'dataset_name:{self.params.dataset.dataset_name} '                    
                    f'run_name:{self.params.common.run_name}')
            print(f'- Model: {self.params.training.model_name}')
            print(f'- TargetDevices & Estimated Inference Times (ms): {performance_infer_time_ms_list}')
            print(f'- This model can be compiled for the above device(s).')
            print('---------------------------------------------------------------------')
        #

    def resolve_run_name(self, run_name, model_name):
        if not run_name:
            return ''
        #
        # modify or set any parameters here as required.
        if '{date-time}' in run_name:
            run_name = run_name.replace('{date-time}', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        #
        if '{model_name}' in run_name:
            run_name = run_name.replace('{model_name}', model_name)
        #
        return run_name

    def clear(self):
        pass

    def prepare(self):
        ret_val = True

        # create folders
        os.makedirs(self.params.common.project_path, exist_ok=True)
        os.makedirs(self.params.common.project_run_path, exist_ok=True)

        #####################################################################
        # handle all downloads here
        utils.download_all(self.params)

        #####################################################################
        # prepare for dataset handling (loading, splitting, limiting files etc).
        self.dataset_handling = datasets.DatasetHandling(self.params)
        # actual dataset handling
        if self.params.dataset.enable:
            self.dataset_handling.clear()
            ret_val = self.dataset_handling.run()
        #
        self.params.update(self.dataset_handling.get_params())

        #####################################################################
        # prepare model training
        self.training_target_module = training.get_target_module(self.params.training.training_backend,
                                                              self.params.common.task_type)
        self.model_training = self.training_target_module.ModelTraining(self.params)
        self.params.update(self.model_training.get_params())

        #####################################################################
        # prepare for model compilation
        self.model_compilation = compilation.edgeai_benchmark.ModelCompilation(self.params)
        self.params.update(self.model_compilation.get_params())

        # write out the description of the current run
        run_params_file = self.write_status_file()
        print(f'INFO: ModelMaker - run params is at: {run_params_file}')

        if not ret_val:
            # dataset is incorrect - training could not start. create a training log file and write error.
            self.params.training.log_file_path = os.path.join(self.params.training.training_path, 'run.log')
            os.makedirs(os.path.dirname(self.params.training.log_file_path), exist_ok=True)
            print('\nERROR: ModelMaker - Training could not start (dataset error)')
            with open(self.params.training.log_file_path, 'a') as lfp:
                lfp.write('\nERROR: ModelMaker - Training could not start (dataset error)')
            #
        #

        return ret_val

    def run(self):
        #####################################################################
        # actual model training
        if self.params.training.enable:
            self.model_training.clear()
            print(f'INFO: ModelMaker - running training - for detailed info see the log file: {self.params.training.log_file_path}')
            self.model_training.run()
            # remove special characters
            utils.cleanup_special_chars(self.params.training.log_file_path)
            # training frameworks don't create a compact package after training. do it here.
            model_training_package_files = [
                self.params.dataset.annotation_path_splits,
                self.params.training.model_checkpoint_path, self.params.training.model_export_path,
                self.params.training.model_proto_path, self.params.training.log_file_path]
            self.package_trained_model(model_training_package_files, self.params.training.model_packaged_path)
            # we are done with training
            print(f'Trained model is at: {self.params.training.training_path}', flush=True)
            print('\nSUCCESS: ModelMaker - Training completed.')
            with open(self.params.training.log_file_path, 'a') as lfp:
                lfp.write('\nSUCCESS: ModelMaker - Training completed.')
            #
        #

        #####################################################################
        # actual model compilation
        if self.params.compilation.enable:
            self.model_compilation.clear()
            print(f'INFO: ModelMaker - running compilation - for detailed info see the log file: {self.params.compilation.log_file_path}')
            self.model_compilation.run()
            print(f'Compiled model is at: {self.params.compilation.model_packaged_path}', flush=True)
            with open(self.params.compilation.log_file_path, 'a') as lfp:
                lfp.write('\nSUCCESS: ModelMaker - Compilation completed.')
            #
        #
        return self.params

    def get_params(self):
        return self.params

    def write_status_file(self):
        run_params_file = os.path.join(self.params.common.project_run_path, 'run.yaml')
        utils.write_dict(self.params, run_params_file)
        # create or update the status file
        status_params_file = os.path.join(self.params.common.project_run_path, 'status.yaml')
        status_params = dict()
        if os.path.exists(status_params_file):
            with open(status_params_file) as fp:
                status_params = yaml.safe_load(fp)
            #
        #
        status_params = utils.ConfigDict(status_params)
        # format the run_params to create status_params
        run_params_formatted = copy.deepcopy(self.params)
        run_params_formatted.compilation = {self.params.common.target_device:run_params_formatted.compilation}
        run_params_formatted = utils.ConfigDict(run_params_formatted)
        status_params.update(run_params_formatted)
        utils.write_dict(status_params, status_params_file)
        return run_params_file

    def package_trained_model(self, input_files, tarfile_name):
        tfp = tarfile.open(tarfile_name, 'w:gz', dereference=True)
        for inpf in input_files:
            inpf_list = inpf if isinstance(inpf, (list,tuple)) else [inpf]
            for inpf_entry in inpf_list:
                if inpf_entry is not None and os.path.exists(inpf_entry):
                    outpf = os.path.basename(inpf_entry)
                    tfp.add(inpf_entry, arcname=outpf)
                #
            #
        #
        tfp.close()
        tarfile_size = os.path.getsize(tarfile_name)
        return tarfile_size


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

    @staticmethod
    def get_version_descriptions(*args, **kwargs):
        return descriptions.get_version_descriptions(*args, **kwargs)

    @staticmethod
    def get_tooltip_descriptions(*args, **kwargs):
        return descriptions.get_tooltip_descriptions(*args, **kwargs)

    @staticmethod
    def get_help_descriptions(*args, **kwargs):
        return descriptions.get_help_descriptions(*args, **kwargs)