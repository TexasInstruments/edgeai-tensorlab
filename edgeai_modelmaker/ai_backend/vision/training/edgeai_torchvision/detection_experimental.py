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
import shutil
import json

from ... import constants
from ... import utils
from references.detection import train


edgeai_modelzoo_path = os.path.abspath('../edgeai-modelzoo')

_pretrained_models = {
    'ssdlite_mobilenet_v2_fpn_lite_tv': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        training=dict(
            training_backend='edgeai_torchvision',
            model_id='od-8160',
            model_name='ssdlite_mobilenet_v2_fpn_lite',
            model_key='ssdlite_mobilenet_v2_fpn_lite_tv',
            model_architecture='ssd',
            input_resize=(512,512),
            input_cropsize=(512,512),
            pretrained_checkpoint=f'{edgeai_modelzoo_path}/models/vision/detection/coco/edgeai-tv/ssdlite_mobilenet_v2_fpn_lite_512x512_20211108_checkpoint.pth',
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=156)
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        )
    ),
    'ssdlite_regnet_x_800mf_fpn_lite_tv': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        training=dict(
            training_backend='edgeai_torchvision',
            model_id='od-8170',
            model_name='ssdlite_regnet_x_800mf_fpn_lite',
            model_key='ssdlite_regnet_x_800mf_fpn_lite_tv',
            model_architecture='ssd',
            input_resize=(512,512),
            input_cropsize=(512,512),
            pretrained_checkpoint_path=f'{edgeai_modelzoo_path}/models/vision/detection/coco/edgeai-tv/ssdlite_regnet_x_800mf_fpn_lite_20211030_checkpoint.pth',
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=103)
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        )
    )
}


def get_pretrained_models(task_type=None):
    if task_type is not None:
        pretrained_models_selected = {k:v for k, v in _pretrained_models.items() if v['task_type'] == task_type}
    else:
        pretrained_models_selected = _pretrained_models
    #
    return pretrained_models_selected


def get_pretrained_model(model_key):
    pretrained_models = get_pretrained_models()
    return pretrained_models[model_key] if model_key in pretrained_models else None


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
        self.train_ann_file = f'{self.params.dataset.dataset_path}/annotations/instances_train.json'
        self.val_ann_file = f'{self.params.dataset.dataset_path}/annotations/instances_val.json'
        with open(self.train_ann_file) as train_ann_fp:
            train_anno = json.load(train_ann_fp)
            categories = train_anno['categories']
            self.object_categories = [cat['name'] for cat in categories]
        #

        # update params that are specific to this backend and model
        self.params.update(
            training=utils.ConfigDict(
                model_checkpoint_path=os.path.join(self.params.training.training_path, 'checkpoint.pth'),
                model_export_path=os.path.join(self.params.training.training_path, 'model-proto.onnx'),
                model_proto_path=os.path.join(self.params.training.training_path, 'model-proto.prototxt'),
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
        device = 'cuda' if self.params.training.num_gpus > 0 else 'cpu'
        # training params
        argv = ['--model', f'{self.params.training.model_name}',
                '--pretrained', f'{self.params.training.pretrained_checkpoint_path}',
                '--dataset', 'modelmaker',
                '--data-path', f'{self.params.dataset.dataset_path}',
                '--num-classes', f'{self.params.training.num_classes}',
                '--gpus', f'{self.params.training.num_gpus}',
                '--output-dir', f'{self.params.training.training_path}',
                '--epochs', f'{self.params.training.training_epochs}',
                '--batch-size', f'{self.params.training.batch_size}',
                '--lr', f'{self.params.training.learning_rate}',
                '--lr-warmup-epochs', f'{self.params.training.warmup_epochs}',
                '--weight-decay', f'{self.params.training.weight_decay}',
                '--distributed', 'True',
                '--device', f'{device}',
                #'--tensorboard-logger', 'True',
                ]
        input_size = self.params.training.input_cropsize if isinstance(self.params.training.input_cropsize, (list,tuple)) else \
            (self.params.training.input_cropsize,self.params.training.input_cropsize)
        argv += ['--input-size', f'{input_size[0]}', f'{input_size[1]}']
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
