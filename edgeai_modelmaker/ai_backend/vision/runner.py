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
from .. import utils
from . import datasets
from . import training
from . import compilation


class ModelRunner():
    @classmethod
    def init_params(self, *args, **kwargs):
        params = dict(
            common=dict(
                verbose_mode=True,
                projects_path='./data/projects',
                project_path=None,
                task_type=None,
                target_device=None,
                run_name=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            ),
            dataset=dict(
                enable=True,
                dataset_name=None,
                dataset_path=None, # dataset split will be created here
                split_factor=0.75,
                split_names=('train', 'val'),
                max_num_files=None,
                input_data_path=None, # input images
                input_annotation_path=None, # annotation file
                data_path_splits=None,
                annotation_path_splits=None,
                annotation_dir='annotations',
                annotation_prefix=None,
                annotation_format='labelstudio_json_min',
                dataset_download=True,
                dataset_reload=False
            ),
            training=dict(
                enable=True,
                model_key=None,
                training_backend=None,
                model_name=None,
                model_id=None,
                pretrained_checkpoint_path=None,
                target_devices={},
                project_path=None,
                dataset_path=None,
                training_path=None,
                checkpoint_path=None,
                model_checkpoint_path=None,
                model_export_path=None,
                model_proto_path=None,
                training_epochs=30,
                warmup_epochs=1,
                num_last_epochs=5,
                batch_size=8,
                learning_rate=2e-3,
                num_classes=None,
                weight_decay=1e-4,
                input_resize=(512,512),
                input_cropsize=(512,512),
                training_device=None, #'cpu', 'cuda'
                num_gpus=0, #0,1..4
                distributed=True,
                training_master_port=29500
            ),
            compilation=dict(
                enable=True,
                compilation_path=None,
                target_device='pc',
                accuracy_level=1,
                tensor_bits=8,
                calibration_frames=10,
                calibration_iterations=10,
                num_frames=None, # inference frame for accuracy test example: 100
                save_output=True # save inference outputs
            ),
        )
        params = utils.ConfigDict(params, *args, **kwargs)
        return params

    @classmethod
    def get_pretrained_models(self, params):
        if params.training.model_key is not None:
            pretrained_model = training.get_pretrained_model(params.training.model_key)
            pretrained_models = {params.training.model_key: pretrained_model}
        else:
            # populate a good pretrained model for the given task
            pretrained_models = training.get_pretrained_models(task_type=params.common.task_type,
                                                               target_device=params.common.target_device,
                                                               training_device=params.training.training_device)
        #
        return pretrained_models

    @classmethod
    def set_pretrined_model(self, params, pretrained_model):
        assert pretrained_model is not None, f'could not find pretrained model for {params.training.model_key}'
        assert params.common.task_type == pretrained_model['common']['task_type'], \
            f'task_type: {params.common.task_type} does not match the pretrained model'
        # get pretrained model checkpoint and other details
        params.update(pretrained_model)
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
        self.params.dataset.dataset_path = os.path.join(self.params.common.project_path, 'dataset')

        run_folder = self.params.common.run_name if self.params.common.run_name else ''
        self.params.training.training_path = os.path.join(self.params.common.project_path, 'run', run_folder, 'training', self.params.training.model_key)
        self.params.compilation.compilation_path = os.path.join(self.params.common.project_path, 'run', run_folder, 'compilation')

        if self.params.common.target_device in self.params.training.target_devices:
            target_device_data = self.params.training.target_devices[self.params.common.target_device]
            performance_fps = target_device_data['performance_fps']
            print(f'DLModel:{self.params.training.model_key} TargetDevice:{self.params.common.target_device} FPS(Estimate):{performance_fps}')
        #

    def clear(self):
        pass

    def run(self):
        # create folders
        os.makedirs(self.params.common.project_path, exist_ok=True)
        os.makedirs(self.params.dataset.dataset_path, exist_ok=True)

        #####################################################################
        # dataset loading, splitting, limiting files etc.
        dataset_handling = datasets.DatasetHandling(self.params)
        self.params.update(dataset_handling.get_params())
        if self.params.dataset.enable:
            dataset_handling.clear()
            dataset_handling.run()
        #

        #####################################################################
        # model training
        training_backend_module = training.get_backend_module(self.params.training.training_backend,
                                                              self.params.common.task_type)
        model_training = training_backend_module.ModelTraining(self.params)
        self.params.update(model_training.get_params())
        if self.params.training.enable:
            model_training.clear()
            model_training.run()
        #

        #####################################################################
        # model compilation
        model_compilation = compilation.edgeai_benchmark.ModelCompilation(self.params)
        self.params.update(model_compilation.get_params())
        if self.params.compilation.enable:
            model_compilation.clear()
            model_compilation.run()
        #
        return self.params
