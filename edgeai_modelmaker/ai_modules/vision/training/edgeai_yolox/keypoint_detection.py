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
import yaml

from ... import constants
from ..... import utils

this_dir_path = os.path.dirname(os.path.abspath(__file__))
repo_parent_path = os.path.abspath(os.path.join(this_dir_path, '../../../../../../'))

edgeai_yolox_path = os.path.join(repo_parent_path, 'edgeai-yolox')
edgeai_modelzoo_path = os.path.join(repo_parent_path, 'edgeai-modelzoo')
www_modelzoo_path = 'https://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/08_05_00_01/edgeai-yolov5/pretrained_models'


_model_descriptions = {
    'yolox-s-human-pose-ti-lite': dict(
        common=dict(
            task_type=constants.TASK_TYPE_KEYPOINT_DETECTION,
        ),
        training=dict(
            training_backend='edgeai_yolox',
            model_name='yolox-s-human-pose-ti-lite',
            model_training_id='yolox-s-human-pose-ti-lite',
            model_architecture='yolox',
            input_resize=640,
            input_cropsize=640,
            pretrained_checkpoint_path='/home/a0504871/Desktop/best_ckpt.pth',
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_KEYPOINT_DETECTION],
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=108, performance_infer_time_ms=1000/108,
                                                     accuracy_factor=56.0, accuracy_unit='AP50%', accuracy_factor2=37.4, accuracy_unit2='AP[.5:.95]%')
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
                'advanced_options:output_feature_16bit_names_list': '370, 426, 482, 538'
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
    if os.path.exists(dst_txt_path):
        os.remove(dst_txt_path)
    with open(src_path) as foo:
        gt = json.load(foo)

    gt_annotations = gt['annotations']
    gt_imgs = {}
    for img in gt["images"]:
        gt_imgs[img['id']] = img
        # Add the image to rain or val list
        with open(dst_txt_path, 'a') as foo:
            foo.write('./{}/{}'.format('images', img['file_name']) + '\n')

    duplicate_ids = []
    cls_offset = 1
    for gt_annotation in gt_annotations:
        if gt_annotation['id'] in duplicate_ids:
            continue
        if gt_annotation['category_id'] == 0:
            cls_offset = 0
        gt_bboxes = [gt_ann['bbox'] for gt_ann in gt_annotations if gt_ann['image_id']==gt_annotation['image_id']]
        gt_bboxes_id = [gt_ann['id'] for gt_ann in gt_annotations if gt_ann['image_id']==gt_annotation['image_id']]
        gt_box = gt_annotation['bbox']
        for box, box_id in zip(gt_bboxes, gt_bboxes_id):
            if box_id in duplicate_ids:
                continue
            if box == gt_box and box_id != gt_annotation['id']:
                duplicate_ids.append(box_id)

    for gt_annotation in gt_annotations:
        if gt_annotation['iscrowd']:
            continue
        if gt_annotation['id'] in duplicate_ids:
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
            cls = gt_annotation['category_id'] - cls_offset
            gt_line = cls, *gt_box
            file_ext = file_name.split('.')[-1]
            file_path = os.path.join(dst_path, file_name.replace(file_ext, "txt"))
            with open(file_path, 'a') as foo:
                foo.write(('%g ' * len(gt_line)).rstrip() % gt_line + '\n')


def create_data_dict(dataset, categories):
    data_dict = {
    'path' : dataset.dataset_path,
    'train' : 'train.txt' ,
    'val' : 'val.txt' ,
    'test' : 'test.txt' ,
    'nc': len(categories) ,
    'names': [category['name'] for category in categories]  #Need to check whether the classes need to be inside quote
    }
    filename_yaml = os.path.join(edgeai_yolox_path, 'data', dataset.dataset_name+'.yaml')
    if os.path.exists(filename_yaml):
        os.remove(filename_yaml)
    with open(filename_yaml, 'w') as foo:
        yaml.safe_dump(data_dict, foo,default_flow_style=None)


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
        self.train_ann_file = f'{self.params.dataset.dataset_path}/annotations/{self.params.dataset.annotation_prefix}_{self.params.dataset.split_names[0]}.json'
        self.val_ann_file = f'{self.params.dataset.dataset_path}/annotations/{self.params.dataset.annotation_prefix}_{self.params.dataset.split_names[1]}.json'
        # self.train_ann_file_yolo = os.path.dirname(self.train_ann_file).replace("annotations", "labels")
        # self.val_ann_file_yolo = os.path.dirname(self.val_ann_file).replace("annotations", "labels")
        # if not os.path.exists(self.train_ann_file_yolo):
        #     json2yolo(self.train_ann_file, self.train_ann_file_yolo, split='train')
        #     json2yolo(self.val_ann_file, self.val_ann_file_yolo, split='val')
        #create_data_dict()
        #create txt file for annotation
        with open(self.train_ann_file) as train_ann_fp:
            train_anno = json.load(train_ann_fp)
            categories = train_anno['categories']
            self.num_keypoints = len(categories[0]['keypoints'])
            self.object_categories = [cat['name'] for cat in categories]
        #
        # create_data_dict(self.params.dataset, categories)

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
        distributed = self.params.training.num_gpus > 1

        args_yolo = {'name': f'{self.params.training.model_training_id}',
                     'dataset': 'coco_kpts',
                     'devices': self.params.training.num_gpus,
                     'batch-size': self.params.training.batch_size,
                     'fp16': True,
                     'occupy': True,
                     'task': 'human_pose',
                     'ckpt': f'{self.params.training.pretrained_checkpoint_path}',
                     'max_epochs': self.params.training.training_epochs,
                     'visualize': False,
                     'output_dir': self.params.training.training_path,
                     'data_dir': f'{self.params.dataset.dataset_path}',
                     'train_ann': self.train_ann_file,
                     'val_ann': self.val_ann_file,
                     'img_folder_names': self.params.dataset.split_names
                     }

        # import dynamically - force_import every time to avoid clashes with scripts in other repositories
        train = utils.import_file_or_folder(os.path.join(edgeai_yolox_path, 'tools', 'train'), __name__, force_import=True)
        export = utils.import_file_or_folder(os.path.join(edgeai_yolox_path, 'tools', 'export_onnx'), __name__, force_import=True)

        # launch the training
        train.run(name=args_yolo['name'],
                  dataset=args_yolo['dataset'],
                  devices=args_yolo['devices'],
                  batch_size=args_yolo['batch-size'],
                  fp16=args_yolo['fp16'],
                  occupy=args_yolo['occupy'],
                  task=args_yolo['task'],
                  max_epochs=args_yolo['max_epochs'],
                  visualize=args_yolo['visualize'],
                  output_dir=args_yolo['output_dir'],
                  data_dir=args_yolo['data_dir'],
                  train_ann=args_yolo['train_ann'],
                  val_ann=args_yolo['val_ann'],
                  img_folder_names=args_yolo['img_folder_names']
        )

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
                            'max_epochs': self.params.training.training_epochs,
                            'output_dir': self.params.training.training_path,
                            'task': 'human_pose',
                            'train_ann': self.train_ann_file,
                            'val_ann': self.val_ann_file
                            }

        # #launch export
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
            max_epochs=args_yolo_export['max_epochs'],
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
