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
import numpy as np

from ... import constants
from ..... import utils


edgeai_yolov5_path = os.path.abspath('../edgeai-yolov5')
edgeai_modelzoo_path = os.path.abspath('../edgeai-modelzoo')
www_modelzoo_path = 'https://software-dl.ti.com/jacinto7/esd/modelzoo/latest'


sys.path.insert(0, edgeai_yolov5_path)
import train
sys.path.pop(0)


_model_descriptions = {
    'yolov5s6_640_ti_lite': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        training=dict(
            training_backend='edgeai_yolov5',
            model_name='yolov5s6_640_ti_lite',
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
            model_name='yolov5m6_640_ti_lite',
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
            model_name='yolov5l6_640_ti_lite',
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


def json2yolo(src_path, dst_path, split='train'):
    """
    Convert the json file to txt file per image as expected by YOLOv5 training framework
    """
    os.makedirs(dst_path, exist_ok=True)
    dst_txt_path = os.path.join(dst_path, '..', split + ".txt")
    with open(src_path) as foo:
        gt = json.load(foo)

    gt_annotations = gt['annotations']
    gt_imgs = {}
    for img in gt["images"]:
        gt_imgs[img['id']] = img
        with open(dst_txt_path, 'a') as foo:
            foo.write('./{}/{}'.format('images', img['file_name']) + '\n')

    for gt_annotation in gt_annotations:
        if gt_annotation['iscrowd']:
            continue

        img = gt_imgs[gt_annotation['image_id']]
        height =img['height']
        width = img['width']
        file_name = img['file_name']
        gt_box = np.array(gt_annotation['bbox'], dtype=np.float64)
        gt_box[:2] += gt_box[2:]/2
        gt_box[[0, 2]] /= width
        gt_box[[1, 3]] /= height
        
        if gt_box[2] >0  and gt_box[3]>0:
            cls = gt_annotation['category_id'] - 1
            gt_line = cls, *gt_box
            file_path = os.path.join(dst_path, file_name.replace("jpg", "txt"))
            with open(file_path, 'a') as foo:
                foo.write(('%g ' * len(gt_line)).rstrip() % gt_line + '\n')


# def create_data_dict(params):
#         data_dict = {
#         'path' : '' ,
#         'train' : '' ,
#         'val' : '' ,
#         'nc': '' ,
#         'names': []
#         }
#
#         dict_obj = simplify_dict(dict_obj)
#         filename_yaml = os.path.splitext(filename)[0] + '.yaml'
#         with open(filename_yaml, 'w') as fp:
#             yaml.safe_dump(dict_obj, fp)



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
        self.train_ann_file_yolo = os.path.dirname(self.train_ann_file).replace("annotations", "labels")
        self.val_ann_file_yolo = os.path.dirname(self.val_ann_file).replace("annotations", "labels")
        json2yolo(self.train_ann_file, self.train_ann_file_yolo, split='train')
        json2yolo(self.val_ann_file, self.val_ann_file_yolo, split='val')
        #create_data_dict()
        #create txt file for annotation
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
        if self.params.training.num_gpus > 0:
            devices = list(range(0, self.params.training.num_gpus))
            devices = [str(device) for device in devices]
            devices = ",".join(devices)
        else:
            devices = "cpu"
        #--data  coco.yaml --cfg yolov5s6.yaml --weights '' --batch - size 63
        # training params
        hyp_path = os.path.join(edgeai_yolov5_path, 'data', 'hyps', 'hyps.scratch.yaml')
        devices = [range(self.params.training.num_gpus)]

        argv = ['--cfg', f'{self.params.training.model_training_id}.yaml',
                '--weights', f'{self.params.training.pretrained_checkpoint_path}',
                '--data', f'{self.params.dataset.dataset_name}.yaml',  #This needs to be written for each dataset
                '--devices', f'{devices}',
                '--output-dir', f'{self.params.training.training_path}',
                '--epochs', f'{self.params.training.training_epochs}',
                '--batch-size', f'{self.params.training.batch_size}',
                '--img', f'{640}',
                '--hyp' f'{hyp_path}',
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
