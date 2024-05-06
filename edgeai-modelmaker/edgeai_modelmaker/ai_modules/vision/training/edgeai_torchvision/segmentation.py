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

this_dir_path = os.path.dirname(os.path.abspath(__file__))
repo_parent_path = os.path.abspath(os.path.join(this_dir_path, '../../../../../../'))

edgeai_torchvision_path = os.path.join(repo_parent_path, 'edgeai-torchvision')
edgeai_modelzoo_path = os.path.join(repo_parent_path, 'edgeai-modelzoo')
www_modelzoo_path = 'https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01'

# Need to modify later based on the available models
_model_descriptions = {
    'fpn_aspp_regnetx800mf_edgeailite': dict(
        common=dict(
            task_type=constants.TASK_TYPE_SEGMENTATION,
        ),
        training=dict(
            training_backend='edgeai_torchvision',
            model_training_id='fpn_aspp_regnetx800mf_edgeailite',
            model_name='fpn_aspp_regnetx800mf_edgeailite',
            model_architecture='backbone',
            input_resize=(512,512),
            input_cropsize=(512,512),
            pretrained_checkpoint_path=f'{www_modelzoo_path}/models/vision/segmentation/cocoseg21/edgeai-tv/fpn_aspp_regnetx800mf_edgeailite_512x512_20210405_checkpoint.pth',
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_SEGMENTATION],
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=272, performance_infer_time_ms=1000/272,
                                                     accuracy_factor=75.212, accuracy_unit='MeanIoU%'),
                # constants.TARGET_DEVICE_AM62A: dict(performance_fps=272, performance_infer_time_ms=3*1000/272,
                #                                      accuracy_factor=75.212, accuracy_unit='MeanIoU%'),
                constants.TARGET_DEVICE_AM68A: dict(performance_fps=272, performance_infer_time_ms=1000/272,
                                                     accuracy_factor=75.212, accuracy_unit='MeanIoU%'),
                constants.TARGET_DEVICE_AM69A: dict(performance_fps=272, performance_infer_time_ms='3.68 (with 1/4th device capability)',
                                                     accuracy_factor=75.212, accuracy_unit='MeanIoU%'),
                constants.TARGET_DEVICE_AM62: dict(performance_fps=272, performance_infer_time_ms=100*1000/272,
                                                     accuracy_factor=75.212, accuracy_unit='MeanIoU%'),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='ss-8720',
            metric=dict(label_offset_pred=1),
            runtime_options=dict(tensor_bits=16),
        )
    ),
    'unet_aspp_mobilenetv2_tv_edgeailite': dict(
        common=dict(
            task_type=constants.TASK_TYPE_SEGMENTATION,
        ),
        training=dict(
            training_backend='edgeai_torchvision',
            model_training_id='unet_aspp_mobilenetv2_tv_edgeailite',
            model_name='unet_aspp_mobilenetv2_tv_edgeailite',
            model_architecture='backbone',
            input_resize=(512,512),
            input_cropsize=(512,512),
            pretrained_checkpoint_path=f'{www_modelzoo_path}/models/vision/segmentation/cocoseg21/edgeai-tv/unet_aspp_mobilenetv2_edgeailite_512x512_20210407_checkpoint.pth',
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_SEGMENTATION]//2,
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=237, performance_infer_time_ms=1000/237,
                                                     accuracy_factor=77.040, accuracy_unit='MeanIoU%'),
                # constants.TARGET_DEVICE_AM62A: dict(performance_fps=237, performance_infer_time_ms=3*1000/237,
                #                                      accuracy_factor=77.040, accuracy_unit='MeanIoU%'),
                constants.TARGET_DEVICE_AM68A: dict(performance_fps=237, performance_infer_time_ms=1000/237,
                                                     accuracy_factor=77.040, accuracy_unit='MeanIoU%'),
                constants.TARGET_DEVICE_AM69A: dict(performance_fps=237, performance_infer_time_ms='4.22 (with 1/4th device capability)',
                                                     accuracy_factor=77.040, accuracy_unit='MeanIoU%'),
                constants.TARGET_DEVICE_AM62: dict(performance_fps=237, performance_infer_time_ms=100*1000/237,
                                                     accuracy_factor=77.040, accuracy_unit='MeanIoU%'),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='ss-8630',
            metric=dict(label_offset_pred=1),
            runtime_options=dict(tensor_bits=8),
        )
    ),
    'deeplabv3plus_mobilenetv2_tv_edgeailite': dict(
        common=dict(
            task_type=constants.TASK_TYPE_SEGMENTATION,
        ),
        training=dict(
            training_backend='edgeai_torchvision',
            model_training_id='deeplabv3plus_mobilenetv2_tv_edgeailite',
            model_name='deeplabv3plus_mobilenetv2_tv_edgeailite',
            model_architecture='backbone',
            input_resize=(512,512),
            input_cropsize=(512,512),
            pretrained_checkpoint_path=f'{www_modelzoo_path}/models/vision/segmentation/cocoseg21/edgeai-tv/deeplabv3plus_mobilenetv2_edgeailite_512x512_20210405_checkpoint.pth',
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_SEGMENTATION]//2,
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=237, performance_infer_time_ms=1000/237,
                                                     accuracy_factor=77.040, accuracy_unit='MeanIoU%'),
                constants.TARGET_DEVICE_AM62A: dict(performance_fps=237, performance_infer_time_ms=3*1000/237,
                                                     accuracy_factor=77.040, accuracy_unit='MeanIoU%'),
                constants.TARGET_DEVICE_AM68A: dict(performance_fps=237, performance_infer_time_ms=1000/237,
                                                     accuracy_factor=77.040, accuracy_unit='MeanIoU%'),
                constants.TARGET_DEVICE_AM69A: dict(performance_fps=237, performance_infer_time_ms='4.22 (with 1/4th device capability)',
                                                     accuracy_factor=77.040, accuracy_unit='MeanIoU%'),
                constants.TARGET_DEVICE_AM62: dict(performance_fps=237, performance_infer_time_ms=100*1000/237,
                                                     accuracy_factor=77.040, accuracy_unit='MeanIoU%'),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='ss-8710',
            metric=dict(label_offset_pred=1),
            runtime_options=dict(tensor_bits=8),
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
            self.params.training.num_classes = len(categories)
            self.object_categories = [cat['name'] for cat in categories]
        #

        log_summary_regex = {
            'js': [
                {'type':'Epoch', 'name':'Epoch', 'description':'Epochs', 'unit':'Epoch', 'value':None,
                 'regex':[{'op':'search', 'pattern':r'Epoch:\s+\[(?<eid>\d+)\]s+Total" -> "validation 100.00%.*Epoch=(?<eid>\d+)/', 'group':1}],
                },
                {'type':'Validation Accuracy', 'name':'Accuracy', 'description':'Validation Accuracy', 'unit':'Accuracy MeanIoU%', 'value':None,
                 'regex':[{'op':'search', 'pattern':r'validation 100%.*MeanIoU=(?<accuracy>[-+e\d+\.\d+]+)', 'group':1, 'scale_factor':1}],
                 }]
        }

        # update params that are specific to this backend and model
        self.params.update(
            training=utils.ConfigDict(
                log_file_path=os.path.join(self.params.training.training_path, 'run.log'),
                log_summary_regex=log_summary_regex,
                summary_file_path=os.path.join(self.params.training.training_path, 'summary.yaml'),
                model_checkpoint_path=os.path.join(self.params.training.training_path, 'checkpoint.pth'),
                model_export_path=os.path.join(self.params.training.training_path, 'model_best.onnx'),
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

        distributed = 1 if self.params.training.num_gpus > 1 else 0
        device = 'cuda' if self.params.training.num_gpus > 0 else 'cpu'

        # training params
        argv = ['--model_name', f'{self.params.training.model_training_id}',
                '--pretrained', f'{self.params.training.pretrained_checkpoint_path}',
                '--dataset', 'common_segmentation',
                '--data_path', f'{self.params.dataset.dataset_path}',
                '--annotation_prefix', f'{self.params.dataset.annotation_prefix}',
                # '--num_classes', f'{self.params.training.num_classes}',
                '--gpus', f'{self.params.training.num_gpus}',
                '--output_dir', f'{self.params.training.training_path}',
                '--epochs', f'{self.params.training.training_epochs}',
                '--batch_size', f'{self.params.training.batch_size}',
                '--lr', f'{self.params.training.learning_rate}',
                '--weight_decay', f'{self.params.training.weight_decay}',
                '--warmup_epochs', '1',
                '--distributed', f'{distributed}',
                '--device', f'{device}',
                '--save_path', f'{self.params.training.training_path}',
                #'--tensorboard-logger', 'True',
                ]
        #input_size = self.params.training.input_cropsize if isinstance(self.params.training.input_cropsize, (list,tuple)) else \
        #    (self.params.training.input_cropsize,self.params.training.input_cropsize)
        #argv += ['--input-size', f'{input_size[0]}', f'{input_size[1]}']
        # import dynamically - force_import every time to avoid clashes with scripts in other repositories
        train = utils.import_file_or_folder(os.path.join(edgeai_torchvision_path,'references', 'edgeailite', 'main', 'pixel2pixel', 'train_segmentation_main.py'), __name__, force_import=True)
        args = train.get_args_parser().parse_args(argv)
        #args.quit_event = self.quit_event
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
