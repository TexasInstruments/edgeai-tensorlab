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
from ..... import utils

this_dir_path = os.path.dirname(os.path.abspath(__file__))
repo_parent_path = os.path.abspath(os.path.join(this_dir_path, '../../../../../../'))

edgeai_yolox_path = os.path.join(repo_parent_path, 'edgeai-yolox')
edgeai_modelzoo_path = os.path.join(repo_parent_path, 'edgeai-modelzoo')
www_modelzoo_path = 'https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01'


model_urls = {
    'yolox_s_keypoint': [
        {
            'download_url': f'{www_modelzoo_path}/models/vision/keypoint/coco/edgeai-yolox/yolox_s_pose_ti_lite_640_20220301_checkpoint.pth',
            'download_path': os.path.join('{download_path}', 'pretrained', 'yolox_s_keypoint')
        },
    ],
}


_model_descriptions = {
    'yolox_s_keypoint': dict(
        common=dict(
            task_type=constants.TASK_TYPE_KEYPOINT_DETECTION,
        ),
        download=model_urls['yolox_s_keypoint'],
        training=dict(
            training_backend='edgeai_yolox',
            model_name='yolox_s_keypoint',
            model_training_id='yolox-s-human-pose-ti-lite',
            model_architecture='yolox',
            input_resize=640,
            input_cropsize=640,
            pretrained_checkpoint_path=f'{www_modelzoo_path}/models/vision/keypoint/coco/edgeai-yolox/yolox_s_pose_ti_lite_640_20220301_checkpoint.pth',
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_KEYPOINT_DETECTION],
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=77.8, performance_infer_time_ms=1000/77.8,
                                                     accuracy_factor=78.0, accuracy_unit='AP50%', accuracy_factor2=49.6, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM68A: dict(performance_fps=68.89, performance_infer_time_ms=1000/68.89,
                                                     accuracy_factor=78.0, accuracy_unit='AP50%', accuracy_factor2=49.6, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM69A: dict(performance_fps=68.89, 
                                                    performance_infer_time_ms=f'{1000/68.89} (with 1/4th device capability)',
                                                    accuracy_factor=78.0, accuracy_unit='AP50%', accuracy_factor2=49.6, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM67A: dict(performance_fps=15.17, performance_infer_time_ms=f'{1000 / 15.17} (with 1/2 device capability)',
                                                    accuracy_factor=78.0, accuracy_unit='AP50%', accuracy_factor2=49.6,
                                                    accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM62A: dict(performance_fps=15.17, performance_infer_time_ms=1000/15.17,
                                                     accuracy_factor=78.0, accuracy_unit='AP50%', accuracy_factor2=49.6, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM62: dict(performance_fps=77.8/200, performance_infer_time_ms=1000/(77.8/200),
                                                     accuracy_factor=78.0, accuracy_unit='AP50%', accuracy_factor2=49.6, accuracy_unit2='AP[.5:.95]%')
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='kd-7060',
            input_optimization=False,
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': '/0/head/cls_preds.0/Conv_output_0, /0/head/reg_preds.0/Conv_output_0, /0/head/obj_preds.0/Conv_output_0, /0/head/kpts_preds.0/Conv_output_0, /0/head/cls_preds.1/Conv_output_0, /0/head/reg_preds.1/Conv_output_0, /0/head/obj_preds.1/Conv_output_0, /0/head/kpts_preds.1/Conv_output_0, /0/head/cls_preds.2/Conv_output_0, /0/head/reg_preds.2/Conv_output_0, /0/head/obj_preds.2/Conv_output_0, /0/head/kpts_preds.2/Conv_output_0'
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

        # train & val annotations file path
        self.train_ann_file = f'{self.params.dataset.dataset_path}/annotations/{self.params.dataset.annotation_prefix}_{self.params.dataset.split_names[0]}.json'
        self.val_ann_file = f'{self.params.dataset.dataset_path}/annotations/{self.params.dataset.annotation_prefix}_{self.params.dataset.split_names[1]}.json'

        # num_keypoints & num_classes collection
        with open(self.train_ann_file) as train_ann_fp:
            train_anno = json.load(train_ann_fp)
            categories = train_anno['categories']
            self.num_keypoints = len(categories[0]['keypoints'])
            self.object_categories = [cat['name'] for cat in categories]
        #

        # regex expressions for logging usage
        log_summary_regex = {
            'js' : [
            {'type':'epoch', 'name':'Epoch', 'description':'Epochs', 'unit':'Epoch', 'value':None,
             'regex':[{'op':'search', 'pattern':r'\s+(\d+),.+', 'group':1}],
            },
            {'type':'training_loss', 'name':'Loss', 'description':'Training Loss', 'unit':'Loss', 'value':None,
             'regex':[{'op':'search', 'pattern':r'TODO-Loss-TODO'}],
             },
            {'type':'validation_accuracy', 'name':'Accuracy', 'description':'Validation Accuracy', 'unit':'AP50%', 'value':None,
             'regex':[{'op':'search', 'pattern':r'\s+[-+e\d+\.\d+]+,\s+[-+e\d+\.\d+]+,\s+[-+e\d+\.\d+]+,\s+[-+e\d+\.\d+]+,'
                                                r'\s+[-+e\d+\.\d+]+,\s+[-+e\d+\.\d+]+,\s+([-+e\d+\.\d+]+)', 'group':1, 'scale_factor':100}],
             }]
        }

        # update params that are specific to this backend and model
        self.params.update(
            training=utils.ConfigDict(
                log_file_path=os.path.join(self.params.training.training_path, 'run.log'),
                log_summary_regex=log_summary_regex,
                summary_file_path=os.path.join(self.params.training.training_path, 'summary.yaml'),
                model_checkpoint_path=os.path.join(self.params.training.training_path, 'best_ckpt.pth'),
                model_export_path=os.path.join(self.params.training.training_path, 'best_model.onnx'),
                model_proto_path=os.path.join(self.params.training.training_path, 'best_model.prototxt'),
                num_classes=len(self.object_categories),
                num_keypoints=self.num_keypoints
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

        # parameters required for the training part
        args_yolo = {'name': f'{self.params.training.model_training_id}',
                     'dataset': 'coco_kpts',
                     'devices': self.params.training.num_gpus,
                     'device_type': 'cuda' if self.params.training.num_gpus else 'cpu',
                     'batch-size': self.params.training.batch_size,
                     'fp16': True if self.params.training.num_gpus else False,
                     'occupy': True if self.params.training.num_gpus else False,
                     'task': 'human_pose',
                     'ckpt': f'{self.params.training.pretrained_checkpoint_path}',
                     'max_epoch': self.params.training.training_epochs,
                     'visualize': False,
                     'output_dir': self.params.training.training_path,
                     'data_dir': f'{self.params.dataset.dataset_path}',
                     'train_ann': self.train_ann_file,
                     'val_ann': self.val_ann_file,
                     'img_folder_names': self.params.dataset.split_names,
                     }

        # import dynamically - force_import every time to avoid clashes with scripts in other repositories
        train = utils.import_file_or_folder(os.path.join(edgeai_yolox_path, 'tools', 'train'), __name__, force_import=True)
        export = utils.import_file_or_folder(os.path.join(edgeai_yolox_path, 'tools', 'export_onnx'), __name__, force_import=True)

        # launch the training
        train.run(name=args_yolo['name'],
                  ckpt=args_yolo['ckpt'],
                  dataset=args_yolo['dataset'],
                  devices=args_yolo['devices'],
                  device_type=args_yolo['device_type'],
                  batch_size=args_yolo['batch-size'],
                  fp16=args_yolo['fp16'],
                  occupy=args_yolo['occupy'],
                  task=args_yolo['task'],
                  max_epoch=args_yolo['max_epoch'],
                  visualize=args_yolo['visualize'],
                  output_dir=args_yolo['output_dir'],
                  data_dir=args_yolo['data_dir'],
                  train_ann=args_yolo['train_ann'],
                  val_ann=args_yolo['val_ann'],
                  img_folder_names=args_yolo['img_folder_names'],
        )

        # parameters required for the compilation part
        args_yolo_export = {'output_name': f'{self.params.training.model_export_path}',
                            'ckpt': None,
                            'name': f'{self.params.training.model_training_id}',
                            'export_det': True,
                            'output': 'yolox_out',
                            'input': 'yolox_in',
                            'batch_size': self.params.training.batch_size,
                            'dataset': 'coco_kpts',
                            'dynamic': False, #True,
                            'opset': 11,
                            'no_onnxsim': False,
                            'max_epoch': self.params.training.training_epochs,
                            'output_dir': self.params.training.training_path,
                            'task': 'human_pose',
                            'train_ann': self.train_ann_file,
                            'val_ann': self.val_ann_file
                            }

        # launch export
        export.run_export(
            output_name=args_yolo_export['output_name'],
            ckpt=args_yolo_export['ckpt'],
            name=args_yolo_export['name'],
            export_det=args_yolo_export['export_det'],
            output=args_yolo_export['output'],
            input=args_yolo_export['input'],
            batch_size=args_yolo_export['batch_size'],
            dataset=args_yolo_export['dataset'],
            dynamic=args_yolo_export['dynamic'],
            opset=args_yolo_export['opset'],
            no_onnxsim=args_yolo_export['no_onnxsim'],
            max_epoch=args_yolo_export['max_epoch'],
            output_dir=args_yolo_export['output_dir'],
            task=args_yolo_export['task'],
            train_ann=args_yolo_export['train_ann'],
            val_ann=args_yolo_export['val_ann']
        )

        return self.params

    def stop(self):
        if self.quit_event is not None:
            self.quit_event.set()
            return True
        #
        return False

    def get_params(self):
        return self.params
