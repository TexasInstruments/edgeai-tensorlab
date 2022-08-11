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

import os
import sys
import shutil
import json

from ... import constants
from ..... import utils


edgeai_torchvision_path = os.path.abspath('../edgeai-yolov5')
edgeai_modelzoo_path = os.path.abspath('../edgeai-modelzoo')
www_modelzoo_path = 'https://software-dl.ti.com/jacinto7/esd/modelzoo/latest'


sys.path.insert(0, edgeai_yolov5_path)
from references.detection import train
sys.path.pop(0)


_model_descriptions = {
    'yolov5s6_640_ti_lite': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        training=dict(
            training_backend='edgeai_yolov5',
            model_name='yolov5s6-ti-lite',
            model_training_id='yolov5s6',
            model_architecture='yolov5',
            input_resize=640,
            input_cropsize=640,
            pretrained_checkpoint_path=f'',
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=134, accuracy_factor=37.4)
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='od-8100',
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': '370, 680, 990, 1300'
            },
            metric=dict(label_offset_pred=0)
        )
    ),
    'yolov5m6_640_ti_lite': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        training=dict(
            training_backend='edgeai_yolov5',
            model_name='yolov5m6-ti-lite',
            model_training_id='yolov5m6',
            model_architecture='yolov5',
            input_resize=640,
            input_cropsize=640,
            pretrained_checkpoint_path=f'',
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=51, accuracy_factor=44.1)
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='od-8120',
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': '228, 498, 808, 1118, 1428'
            },
            metric=dict(label_offset_pred=0)
        )
    ),
    'yolov5l6_640_ti_lite': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        training=dict(
            training_backend='edgeai_yolov5',
            model_name='yolov5l6-ti-lite',
            model_training_id='yolov5l6',
            model_architecture='yolov5',
            input_resize=640,
            input_cropsize=640,
            pretrained_checkpoint_path=f'',
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=26, accuracy_factor=47.1)
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='od-8130',
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': '288, 626, 936, 1246, 1556'
            },
            metric=dict(label_offset_pred=0)
        )
    ),
}


def get_model_descriptions(task_type=None):
    if task_type is not None:
        model_descriptions_selected = {k:v for k, v in _model_descriptions.items() if v['task_type'] == task_type}
    else:
        model_descriptions_selected = _model_descriptions
    #
    return model_descriptions_selected


def get_model_description(model_name):
    model_descriptions = get_model_descriptions()
    return model_descriptions[model_name] if model_name in model_descriptions else None


class ModelTraining:
    @classmethod
    def init_params(self, *args, **kwargs):
        params = dict(
            training=dict(
            )
        )
        params = utils.ConfigDict(params, *args, **kwargs)
        return params

    def __init__(self, *args, quit_event=None, **kwargs):
        self.params = self.init_params(*args, **kwargs)
        self.quit_event = quit_event

        # num classes
        self.train_ann_file = f'{self.params.dataset.dataset_path}/annotations/{self.params.dataset.annotation_prefix}_train.json'
        self.val_ann_file = f'{self.params.dataset.dataset_path}/annotations/{self.params.dataset.annotation_prefix}_val.json'
        with open(self.train_ann_file) as train_ann_fp:
            train_anno = json.load(train_ann_fp)
            categories = train_anno['categories']
            self.object_categories = [cat['name'] for cat in categories]
        #

        # update params that are specific to this backend and model
        self.params.update(
            training=utils.ConfigDict(
                log_file_path=os.path.join(self.params.training.training_path, 'run.log'),
                checkpoint_path=os.path.join(self.params.training.training_path, 'checkpoint.pth'),
                model_export_path=os.path.join(self.params.training.training_path, 'model.onnx'),
                model_proto_path=None,
                num_classes=len(self.object_categories),
            )
        )

    def clear(self):
        # clear the training folder
        shutil.rmtree(self.params.training.training_path, ignore_errors=True)

    def run(self, **kwargs):
        ''''
        The actual training function. Move this to a worker process, if this function is called from a GUI.
        '''
        os.makedirs(self.params.training.training_path, exist_ok=True)

        distributed = self.params.training.num_gpus > 1
        device = 'cuda' if self.params.training.num_gpus > 0 else 'cpu'
        # training params
        argv = ['--model', f'{self.params.training.model_training_id}',
                '--pretrained', f'{self.params.training.pretrained_checkpoint_path}',
                '--dataset', 'modelmaker',
                '--data-path', f'{self.params.dataset.dataset_path}',
                '--annotation-prefix', f'{self.params.dataset.annotation_prefix}',
                #'--num-classes', f'{self.params.training.num_classes}',
                '--gpus', f'{self.params.training.num_gpus}',
                '--output-dir', f'{self.params.training.training_path}',
                '--epochs', f'{self.params.training.training_epochs}',
                '--batch-size', f'{self.params.training.batch_size}',
                '--lr', f'{self.params.training.learning_rate}',
                '--weight-decay', f'{self.params.training.weight_decay}',
                '--lr-warmup-epochs', '1',
                '--distributed', f'{distributed}',
                '--device', f'{device}',
                #'--tensorboard-logger', 'True',
                ]
        #input_size = self.params.training.input_cropsize if isinstance(self.params.training.input_cropsize, (list,tuple)) else \
        #    (self.params.training.input_cropsize,self.params.training.input_cropsize)
        #argv += ['--input-size', f'{input_size[0]}', f'{input_size[1]}']
        args = train.get_args_parser().parse_args(argv)
        args.quit_event = self.quit_event
        # launch the training
        train.run(args)

        return self.params

    def stop(self):
        if self.quit_event is not None:
            self.quit_event.set()
            return True
        #
        return False

    def get_params(self):
        return self.params
