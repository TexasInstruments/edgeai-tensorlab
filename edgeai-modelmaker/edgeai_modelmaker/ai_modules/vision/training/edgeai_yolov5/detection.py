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

edgeai_yolov5_path = os.path.join(repo_parent_path, 'edgeai-yolov5')
edgeai_modelzoo_path = os.path.join(repo_parent_path, 'edgeai-modelzoo')
www_modelzoo_path = 'https://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/latest/edgeai-yolov5/pretrained_models'


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
            pretrained_checkpoint_path=f'{www_modelzoo_path}/checkpoints/detection/coco/edgeai-yolov5/yolov5s6_640_ti_lite/weights/best.pt',
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_DETECTION],
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=108, performance_infer_time_ms=1000/108,
                                                     accuracy_factor=37.4, accuracy_unit='AP[.5:.95]%')
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='od-8100',
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': '370, 426, 482, 538'
            },
            metric=dict(label_offset_pred=0)
        )
    ),
    'yolov5s6_384_ti_lite': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        training=dict(
            training_backend='edgeai_yolov5',
            model_name='yolov5s6_384_ti_lite',
            model_training_id='yolov5s6',
            model_architecture='yolov5',
            input_resize=384,
            input_cropsize=384,
            pretrained_checkpoint_path=f'{www_modelzoo_path}/checkpoints/detection/coco/edgeai-yolov5/yolov5s6_384_ti_lite/weights/best.pt',
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_DETECTION],
            target_devices={  #To Update
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=170, performance_infer_time_ms=1000/170,
                                                     accuracy_factor=32.8, accuracy_unit='AP[.5:.95]%')
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='od-8110',
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': '370, 426, 482, 538'
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
            pretrained_checkpoint_path=f'{www_modelzoo_path}/checkpoints/detection/coco/edgeai-yolov5/yolov5m6_640_ti_lite/weights/best.pt',
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_DETECTION]//2,
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=46, performance_infer_time_ms=1000/46,
                                                     accuracy_factor=44.1, accuracy_unit='AP[.5:.95]%')
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='od-8120',
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': '228, 498, 554, 610, 666'
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
            pretrained_checkpoint_path=f'{www_modelzoo_path}/checkpoints/detection/coco/edgeai-yolov5/yolov5l6_640_ti_lite/weights/best.pt',
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_DETECTION]//2,
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=21, performance_infer_time_ms=1000/21,
                                                     accuracy_factor=47.1, accuracy_unit='AP[.5:.95]%')
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='od-8130',
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': '288, 626, 682, 738, 794'
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
    filename_yaml = os.path.join(edgeai_yolov5_path, 'data', dataset.dataset_name+'.yaml')
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
        self.train_ann_file = f'{self.params.dataset.dataset_path}/annotations/{self.params.dataset.annotation_prefix}_train.json'
        self.val_ann_file = f'{self.params.dataset.dataset_path}/annotations/{self.params.dataset.annotation_prefix}_val.json'
        self.train_ann_file_yolo = os.path.dirname(self.train_ann_file).replace("annotations", "labels")
        self.val_ann_file_yolo = os.path.dirname(self.val_ann_file).replace("annotations", "labels")
        if not os.path.exists(self.train_ann_file_yolo):
            json2yolo(self.train_ann_file, self.train_ann_file_yolo, split='train')
            json2yolo(self.val_ann_file, self.val_ann_file_yolo, split='val')
        #create_data_dict()
        #create txt file for annotation
        with open(self.train_ann_file) as train_ann_fp:
            train_anno = json.load(train_ann_fp)
            categories = train_anno['categories']
            self.object_categories = [cat['name'] for cat in categories]
        #
        create_data_dict(self.params.dataset, categories)

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
                model_checkpoint_path=os.path.join(self.params.training.training_path, 'weights','best.pt'),
                model_export_path=os.path.join(self.params.training.training_path, 'weights', 'best.onnx'),
                model_proto_path=os.path.join(self.params.training.training_path,  'weights', 'best.prototxt'),
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
        hyp_path = os.path.join(edgeai_yolov5_path, 'data', 'hyps', 'hyp.scratch.yaml')
        with open(hyp_path) as fp:
            hyp_config = yaml.safe_load(fp)
        hyp_config['lr0'] = self.params.training.learning_rate
        hyp_config['weight_decay'] = self.params.training.weight_decay
        hyp_path = os.path.join(self.params.training.training_path, 'hyp.yaml')
        with open(hyp_path, 'w') as fp:
            yaml.safe_dump(hyp_config, fp)

        device = list(range(self.params.training.num_gpus))
        device = ','.join([str(id) for id in device]) if len(device)>0 else 'cpu'
        project_path = self.params.training['training_path']

        data_yaml_path = os.path.join(edgeai_yolov5_path, 'data', self.params.dataset.dataset_name + '.yaml' )
        with open(data_yaml_path) as fp:
            data_config = yaml.safe_load(fp)
        data_yaml_path = os.path.join(self.params.training.training_path, self.params.dataset.dataset_name + '.yaml' )
        with open(data_yaml_path, 'w') as fp:
            yaml.safe_dump(data_config, fp)

        yolo_cfg = os.path.join(edgeai_yolov5_path, 'models', 'hub', self.params.training.model_training_id + '.yaml')

        args_yolo = {'cfg': f'{yolo_cfg}',
                    'weights': f'{self.params.training.pretrained_checkpoint_path}',
                    'data' : f'{data_yaml_path}',  #This needs to be written for each dataset
                    'device' : f'{device}',
                    'epochs' : self.params.training.training_epochs,
                    'batch-size' : self.params.training.batch_size,
                    'imgsz': self.params.training.input_resize,
                    'hyp': f'{hyp_path}',
                    'project': f'{project_path}',
                    'noautoanchor': False, #Set this to True to disable autoanchor
                    }

        # import dynamically - force_import every time to avoid clashes with scripts in other repositories
        train = utils.import_file_or_folder(os.path.join(edgeai_yolov5_path,'train'), __name__, force_import=True)
        export = utils.import_file_or_folder(os.path.join(edgeai_yolov5_path,'export'), __name__, force_import=True)

        # launch the training
        train.run(cfg=args_yolo['cfg'], weights=args_yolo['weights'], data=args_yolo['data'],
                  device=args_yolo['device'], epochs=args_yolo['epochs'],
                  batch_size=args_yolo['batch-size'], imgsz=args_yolo['imgsz'],
                  hyp=args_yolo['hyp'], project=args_yolo['project'], name='',
                  exist_ok=True, noautoanchor=args_yolo['noautoanchor'],
                  disable_git_status=True)

        args_yolo_export = {
            'weights': self.params.training['model_checkpoint_path'],
            'imgsz': (self.params.training.input_resize, self.params.training.input_resize),
            'simplify': True,
            'batch_size': 1,
            'opset': 11,
            'export-nms': True,
            'include': ['onnx'],
            'simple_search': True
        }
        #launch export
        export.run(weights=args_yolo_export['weights'], img_size=args_yolo_export['imgsz'], simplify=args_yolo_export['simplify'],
                   batch_size=args_yolo_export['batch_size'], opset=args_yolo_export['opset'], export_nms=args_yolo_export['export-nms'],
                   include=args_yolo_export['include'], simple_search=args_yolo_export['simple_search'])


        return self.params

    def stop(self):
        if self.quit_event is not None:
            self.quit_event.set()
            return True
        #
        return False

    def get_params(self):
        return self.params
