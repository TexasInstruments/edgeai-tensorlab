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
import sys
import json
from torch.distributed import run as distributed_run
import subprocess

import edgeai_benchmark
from ... import constants
from ..... import utils

this_dir_path = os.path.dirname(os.path.abspath(__file__))
repo_parent_path = os.path.abspath(os.path.join(this_dir_path, '../../../../../../'))

edgeai_modelzoo_path = os.path.join(repo_parent_path, 'edgeai-modelzoo')
www_modelzoo_path = 'https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01'
edgeai_mmdetection_path = os.path.join(repo_parent_path, 'edgeai-mmdetection')
edgeai_mmdetection_tools_path = os.path.join(edgeai_mmdetection_path, 'tools')


model_urls = {
    # 'yolox_femto_lite': [
    #     {
    #         'download_url': f'{www_modelzoo_path}/models/vision/detection/coco/edgeai-mmdet/yolox_femto_lite_320x320_20230407_checkpoint.pth',
    #         'download_path': os.path.join('{download_path}', 'pretrained', 'yolox_femto_lite')
    #     },
    # ],
    'yolox_pico_lite': [
        {
            'download_url': f'{www_modelzoo_path}/models/vision/detection/coco/edgeai-mmdet/yolox_pico_lite_320x320_20230410_checkpoint.pth',
            'download_path': os.path.join('{download_path}', 'pretrained', 'yolox_pico_lite')
        },
    ],
    'yolox_nano_lite': [
        {
            'download_url': f'{www_modelzoo_path}/models/vision/detection/coco/edgeai-mmdet/yolox_nano_lite_416x416_20220214_checkpoint.pth',
            'download_path': os.path.join('{download_path}', 'pretrained', 'yolox_nano_lite')
        },
    ],
    'yolox_tiny_lite': [
        {
            'download_url': f'{www_modelzoo_path}/models/vision/detection/coco/edgeai-mmdet/yolox_tiny_lite_416x416_20220217_checkpoint.pth',
            'download_path': os.path.join('{download_path}', 'pretrained', 'yolox_tiny_lite')
        },
    ],
    'yolox_s_lite': [
        {
            'download_url': f'{www_modelzoo_path}/models/vision/detection/coco/edgeai-mmdet/yolox_s_lite_640x640_20220221_checkpoint.pth',
            'download_path': os.path.join('{download_path}', 'pretrained', 'yolox_s_lite')
        },
    ],
    'yolov7_l_lite': [
        {
            'download_url': f'{www_modelzoo_path}/models/vision/detection/coco/edgeai-mmdet/yolov7_l_coco_lite_640x640_20250109_checkpoint.pth',
            'download_path': os.path.join('{download_path}', 'pretrained', 'yolov7_l_lite')
        },
    ],
    'yolov9_s_lite': [
        {
            'download_url': f'{www_modelzoo_path}/models/vision/detection/coco/edgeai-mmdet/yolov9_s_coco_lite_640x640_20250219_checkpoint.pth',
            'download_path': os.path.join('{download_path}', 'pretrained', 'yolov9_s_lite')
        },
    ],
}


_model_descriptions = {
    # 'yolox_femto_lite': dict(
    #     common=dict(
    #         task_type=constants.TASK_TYPE_DETECTION,
    #     ),
    #     download=model_urls['yolox_femto_lite'],
    #     training=dict(
    #         training_backend='edgeai_mmdetection',
    #         model_name='yolox_femto_lite',
    #         model_training_id='yolox_femto_lite',
    #         model_architecture='yolox',
    #         input_resize=320,
    #         input_cropsize=320,
    #         pretrained_checkpoint_path=model_urls['yolox_femto_lite'][0],
    #         batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_DETECTION],
    #         target_devices={
    #             constants.TARGET_DEVICE_AM62: dict(performance_fps=None, performance_infer_time_ms=87.13,
    #                                             accuracy_factor=21.9, accuracy_unit='AP50%', accuracy_factor2=12.7, accuracy_unit2='AP[.5:.95]%'),
    #         },
    #         training_devices={
    #             constants.TRAINING_DEVICE_CPU: True,
    #             constants.TRAINING_DEVICE_CUDA: True,
    #         }
    #     ),
    #     compilation=dict(
    #         model_compilation_id='od-8260',
    #         runtime_options={
    #             'advanced_options:output_feature_16bit_names_list': '/multi_level_conv_obj.2/Conv_output_0, /multi_level_conv_reg.2/Conv_output_0, /multi_level_conv_cls.2/Conv_output_0, /multi_level_conv_obj.1/Conv_output_0, /multi_level_conv_reg.1/Conv_output_0, /multi_level_conv_cls.1/Conv_output_0, /multi_level_conv_obj.0/Conv_output_0, /multi_level_conv_reg.0/Conv_output_0, /multi_level_conv_cls.0/Conv_output_0'
    #         },
    #         metric=dict(label_offset_pred=0)
    #     )
    # ),
    'yolox_pico_lite': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        download=model_urls['yolox_pico_lite'],
        training=dict(
            training_backend='edgeai_mmdetection',
            model_name='yolox_pico_lite',
            model_training_id='yolox_pico_lite',
            model_architecture='yolox',
            input_resize=320,
            input_cropsize=320,
            pretrained_checkpoint_path=model_urls['yolox_pico_lite'][0],
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_DETECTION],
            target_devices={
                constants.TARGET_DEVICE_AM62: dict(performance_fps=None, performance_infer_time_ms=154.89,
                                                accuracy_factor=29.4, accuracy_unit='AP50%', accuracy_factor2=17.9, accuracy_unit2='AP[.5:.95]%'),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='od-8270',
            input_optimization=False,
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': '/multi_level_conv_obj.2/Conv_output_0, /multi_level_conv_reg.2/Conv_output_0, /multi_level_conv_cls.2/Conv_output_0, /multi_level_conv_obj.1/Conv_output_0, /multi_level_conv_reg.1/Conv_output_0, /multi_level_conv_cls.1/Conv_output_0, /multi_level_conv_obj.0/Conv_output_0, /multi_level_conv_reg.0/Conv_output_0, /multi_level_conv_cls.0/Conv_output_0'
            },
            metric=dict(label_offset_pred=0)
        )
    ),
    'yolox_nano_lite': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        download=model_urls['yolox_nano_lite'],
        training=dict(
            training_backend='edgeai_mmdetection',
            model_name='yolox_nano_lite',
            model_training_id='yolox_nano_lite',
            model_architecture='yolox',
            input_resize=416,
            input_cropsize=416,
            pretrained_checkpoint_path=model_urls['yolox_nano_lite'][0],
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_DETECTION],
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=None, performance_infer_time_ms=3.74,
                                                     accuracy_factor=40.1, accuracy_unit='AP50%', accuracy_factor2=24.8, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM62A: dict(performance_fps=None, performance_infer_time_ms=8.87,
                                                     accuracy_factor=40.1, accuracy_unit='AP50%', accuracy_factor2=24.8, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM67A: dict(performance_fps=None, performance_infer_time_ms='8.87 (with 1/2 device capability)',
                                                    accuracy_factor=40.1, accuracy_unit='AP50%', accuracy_factor2=24.8,
                                                    accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM68A: dict(performance_fps=None, performance_infer_time_ms=3.73,
                                                     accuracy_factor=40.1, accuracy_unit='AP50%', accuracy_factor2=24.8, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM69A: dict(performance_fps=None, performance_infer_time_ms='3.64 (with 1/4th device capability)',
                                                     accuracy_factor=40.1, accuracy_unit='AP50%', accuracy_factor2=24.8, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM62: dict(performance_fps=None, performance_infer_time_ms=516.15,
                                                     accuracy_factor=40.1, accuracy_unit='AP50%', accuracy_factor2=24.8, accuracy_unit2='AP[.5:.95]%'),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='od-8200',
            input_optimization=False,
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': '/multi_level_conv_obj.2/Conv_output_0, /multi_level_conv_reg.2/Conv_output_0, /multi_level_conv_cls.2/Conv_output_0, /multi_level_conv_obj.1/Conv_output_0, /multi_level_conv_reg.1/Conv_output_0, /multi_level_conv_cls.1/Conv_output_0, /multi_level_conv_obj.0/Conv_output_0, /multi_level_conv_reg.0/Conv_output_0, /multi_level_conv_cls.0/Conv_output_0'
            },
            metric=dict(label_offset_pred=0)
        )
    ),
    'yolox_tiny_lite': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        download=model_urls['yolox_tiny_lite'],
        training=dict(
            training_backend='edgeai_mmdetection',
            model_name='yolox_tiny_lite',
            model_training_id='yolox_tiny_lite',
            model_architecture='yolox',
            input_resize=416,
            input_cropsize=416,
            pretrained_checkpoint_path=model_urls['yolox_tiny_lite'][0],
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_DETECTION],
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=None, performance_infer_time_ms=4.84,
                                                     accuracy_factor=47.4, accuracy_unit='AP50%', accuracy_factor2=30.5, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM62A: dict(performance_fps=None, performance_infer_time_ms=15.35,
                                                     accuracy_factor=47.4, accuracy_unit='AP50%', accuracy_factor2=30.5, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM67A: dict(performance_fps=None, performance_infer_time_ms='15.35 (with 1/2 device capability)',
                                                    accuracy_factor=47.4, accuracy_unit='AP50%', accuracy_factor2=30.5,
                                                    accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM68A: dict(performance_fps=None, performance_infer_time_ms=4.82,
                                                     accuracy_factor=47.4, accuracy_unit='AP50%', accuracy_factor2=30.5, accuracy_unit2='AP[.5:.95]%'), #TODO: this has to be corrected
                constants.TARGET_DEVICE_AM69A: dict(performance_fps=None, performance_infer_time_ms='4.73 (with 1/4th device capability)',
                                                     accuracy_factor=47.4, accuracy_unit='AP50%', accuracy_factor2=30.5, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM62: dict(performance_fps=None, performance_infer_time_ms=1130.6,
                                                     accuracy_factor=47.4, accuracy_unit='AP50%', accuracy_factor2=30.5, accuracy_unit2='AP[.5:.95]%')
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='od-8210',
            input_optimization=False,
            runtime_options={
                # 'advanced_options:output_feature_16bit_names_list': '/multi_level_conv_obj.2/Conv_output_0, /multi_level_conv_reg.2/Conv_output_0, /multi_level_conv_cls.2/Conv_output_0, /multi_level_conv_obj.1/Conv_output_0, /multi_level_conv_reg.1/Conv_output_0, /multi_level_conv_cls.1/Conv_output_0, /multi_level_conv_obj.0/Conv_output_0, /multi_level_conv_reg.0/Conv_output_0, /multi_level_conv_cls.0/Conv_output_0'
            },
            metric=dict(label_offset_pred=0)
        )
    ),
    'yolox_s_lite': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        download=model_urls['yolox_s_lite'],
        training=dict(
            training_backend='edgeai_mmdetection',
            model_name='yolox_s_lite',
            model_training_id='yolox_s_lite',
            model_architecture='yolox',
            input_resize=640,
            input_cropsize=640,
            pretrained_checkpoint_path=model_urls['yolox_s_lite'][0],
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_DETECTION],
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=None, performance_infer_time_ms=10.14,
                                                     accuracy_factor=56.9, accuracy_unit='AP50%', accuracy_factor2=38.3, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM62A: dict(performance_fps=None, performance_infer_time_ms=43.94,
                                                     accuracy_factor=56.9, accuracy_unit='AP50%', accuracy_factor2=38.3, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM67A: dict(performance_fps=None, performance_infer_time_ms='43.94 (with 1/2 device capability)',
                                                    accuracy_factor=56.9, accuracy_unit='AP50%', accuracy_factor2=38.3,
                                                    accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM68A: dict(performance_fps=None, performance_infer_time_ms=10.22,
                                                     accuracy_factor=56.9, accuracy_unit='AP50%', accuracy_factor2=38.3, accuracy_unit2='AP[.5:.95]%'), #TODO: this has to be corrected
                constants.TARGET_DEVICE_AM69A: dict(performance_fps=None, performance_infer_time_ms='9.82 (with 1/4th device capability)',
                                                     accuracy_factor=56.9, accuracy_unit='AP50%', accuracy_factor2=38.3, accuracy_unit2='AP[.5:.95]%'),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='od-8220',
            input_optimization=False,
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': '/multi_level_conv_obj.2/Conv_output_0, /multi_level_conv_reg.2/Conv_output_0, /multi_level_conv_cls.2/Conv_output_0, /multi_level_conv_obj.1/Conv_output_0, /multi_level_conv_reg.1/Conv_output_0, /multi_level_conv_cls.1/Conv_output_0, /multi_level_conv_obj.0/Conv_output_0, /multi_level_conv_reg.0/Conv_output_0, /multi_level_conv_cls.0/Conv_output_0'
            },
            metric=dict(label_offset_pred=0)
        )
    ),
    'yolov7_l_lite': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        download=model_urls['yolov7_l_lite'],
        training=dict(
            training_backend='edgeai_mmdetection',
            model_name='yolov7_l_lite',
            model_training_id='yolov7_l_coco_lite',
            model_architecture='yolov7',
            input_resize=640,
            input_cropsize=640,
            pretrained_checkpoint_path=model_urls['yolov7_l_lite'][0],
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_DETECTION], #TODO: performance_infer_time_ms to be updated
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=None, performance_infer_time_ms=-1,
                                                     accuracy_factor=65.1, accuracy_unit='AP50%', accuracy_factor2=45.9, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM62A: dict(performance_fps=None, performance_infer_time_ms=-1,
                                                     accuracy_factor=65.1, accuracy_unit='AP50%', accuracy_factor2=45.9, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM67A: dict(performance_fps=None, performance_infer_time_ms='-1* (with 1/2 device capability)',
                                                    accuracy_factor=65.1, accuracy_unit='AP50%', accuracy_factor2=45.9,
                                                    accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM68A: dict(performance_fps=None, performance_infer_time_ms=33.56,
                                                     accuracy_factor=65.1, accuracy_unit='AP50%', accuracy_factor2=45.9, accuracy_unit2='AP[.5:.95]%'), #TODO: this has to be corrected
                constants.TARGET_DEVICE_AM69A: dict(performance_fps=None, performance_infer_time_ms='-1* (with 1/4th device capability)',
                                                     accuracy_factor=65.1, accuracy_unit='AP50%', accuracy_factor2=45.9, accuracy_unit2='AP[.5:.95]%'),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='od-9202',
            input_optimization=False,
            metric=dict(label_offset_pred=1)
        )
    ),
    'yolov9_s_lite': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        download=model_urls['yolov9_s_lite'],
        training=dict(
            training_backend='edgeai_mmdetection',
            model_name='yolov9_s_lite',
            model_training_id='yolov9_s_coco_lite',
            model_architecture='yolov9',
            input_resize=640,
            input_cropsize=640,
            pretrained_checkpoint_path=model_urls['yolov9_s_lite'][0],
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_DETECTION], #TODO: performance_infer_time_ms to be updated
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=None, performance_infer_time_ms=-1,
                                                     accuracy_factor=54.0, accuracy_unit='AP50%', accuracy_factor2=38.3, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM62A: dict(performance_fps=None, performance_infer_time_ms=-1,
                                                     accuracy_factor=54.0, accuracy_unit='AP50%', accuracy_factor2=38.3, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM67A: dict(performance_fps=None, performance_infer_time_ms='-1* (with 1/2 device capability)',
                                                    accuracy_factor=54.0, accuracy_unit='AP50%', accuracy_factor2=38.3,
                                                    accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM68A: dict(performance_fps=None, performance_infer_time_ms=-1,
                                                     accuracy_factor=54.0, accuracy_unit='AP50%', accuracy_factor2=38.3, accuracy_unit2='AP[.5:.95]%'), #TODO: this has to be corrected
                constants.TARGET_DEVICE_AM69A: dict(performance_fps=None, performance_infer_time_ms='-1* (with 1/4th device capability)',
                                                     accuracy_factor=54.0, accuracy_unit='AP50%', accuracy_factor2=38.3, accuracy_unit2='AP[.5:.95]%'),
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='od-9204',
            input_optimization=False,
            metric=dict(label_offset_pred=1)
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

        # num_classes
        self.train_ann_file = f'{self.params.dataset.dataset_path}/annotations/{self.params.dataset.annotation_prefix}_train.json'
        self.val_ann_file = f'{self.params.dataset.dataset_path}/annotations/{self.params.dataset.annotation_prefix}_val.json'
        with open(self.train_ann_file) as train_ann_fp:
            train_anno = json.load(train_ann_fp)
            categories = train_anno['categories']
            self.object_categories = [cat['name'] for cat in categories]
        #

        log_summary_regex = {
            'js': [
            {'type':'Epoch', 'name':'Epoch', 'description':'Epochs', 'unit':'Epoch', 'value':None,
             'regex':[{'op':'search', 'pattern':r'Epoch\(.*?\)\s+\[(?<eid>\d+)]', 'group':1}],
             },
            {'type':'Validation Accuracy', 'name':'Accuracy', 'description':'Validation Accuracy%', 'unit':'AP50%', 'value':None,
             'regex':[{'op':'search', 'pattern':r'Epoch\(.*?\)\s+\[(?<eid>\d+)]\[\d+].*?bbox_mAP_50:\s+(?<bbox>\d+\.\d+)', 'group':1, 'dtype':'float', 'scale_factor':100}],
             }]
        }

        # update params that are specific to this backend and model
        self.params.update(
            training=utils.ConfigDict(
                log_file_path=os.path.join(self.params.training.training_path, 'run.log'),
                log_summary_regex=log_summary_regex,
                summary_file_path=os.path.join(self.params.training.training_path, 'summary.yaml'),
                model_checkpoint_path=os.path.join(self.params.training.training_path, 'latest.pth'),
                model_export_path=os.path.join(self.params.training.training_path, 'model.onnx'),
                model_proto_path=os.path.join(self.params.training.training_path, 'model.prototxt'),
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
        task_list = [{
            'proc_name':f'{self.params.common.run_name}:training',
            'proc_func':self._proc_func,
            'proc_log':self.params.training.log_file_path,
            'proc_error':[]
        }]
        task_entries = {self.params.training.model_name:task_list}
        parallel_processes = (1 if self.params.compilation.capture_log else 0)
        process_runner = edgeai_benchmark.utils.ProcessRunner(parallel_processes=parallel_processes)
        process_runner.run(task_entries)
        return self.params

    def _proc_func(self, **kwargs):
        ''''
        The actual training function. Move this to a worker process, if this function is called from a GUI.
        '''
        os.makedirs(self.params.training.training_path, exist_ok=True)
        # training params
        # dataset_style = 'coco' #'voc' #'coco'
        # input_size = self.params.training.input_cropsize if isinstance(self.params.training.input_cropsize, (list,tuple)) else \
        #     (self.params.training.input_cropsize,self.params.training.input_cropsize)
        base_config_path = os.path.join(edgeai_mmdetection_path, 'configs_edgeailite', self.params.training.model_architecture, self.params.training.model_training_id, )
        classes = tuple(self.object_categories)
        config_file = os.path.join(self.params.training.training_path, f'{self.params.training.model_name}.py')
        config_strs = []
        config_strs += [f'_base_   = ['
                        f'"{base_config_path}.py"]\n']
        config_strs += [f'work_dir   = "{self.params.training.training_path}"']
        config_strs += [f'classes = {classes}']
        config_strs += [f'data_root   = "{self.params.dataset.dataset_path}"']
        config_strs += [f'max_epochs   = {self.params.training.training_epochs}']
        config_strs += [f'export_onnx_model=True']
        config_strs += [f'optim_wrapper=dict(\n'
                        f'  optimizer=dict(\n'
                        f'    lr={self.params.training.learning_rate}\n'
                        f'  ) \n'                            
                        f') \n']
        config_strs += [f'model=dict(\n'
                        f'  bbox_head=dict(\n'
                        f'    num_classes={self.params.training.num_classes}\n'
                        f'  ) \n'                            
                        f') \n']
        if 'yolov9' in base_config_path:
            config_strs += [f'model=dict(\n'
                        f'  bbox_head=dict(\n'
                        f'    num_classes={self.params.training.num_classes},\n'
                        f'    loss_yolo=dict(\n'
                        f'      class_num={self.params.training.num_classes},\n'
                        f'   ) \n'
                        f'  ), \n'
                        f'  aux_head=dict(\n'
                        f'    num_classes={self.params.training.num_classes}, \n'
                        f'  ) \n'                             
                        f') \n']
        config_strs += [f'train_dataloader=dict(\n'
                        f'  dataset=dict(\n'
                        f'      dataset=dict(\n'
                        f'      data_root   = "{self.params.dataset.dataset_path}",\n'
                        f'          ann_file = "{self.train_ann_file}",\n'
                        f'          data_prefix = dict(img="train/"),\n'
                        f'          metainfo=dict(classes=classes),\n'
                        f'          ) \n'
                        f'      ) \n'
                        f'  ) \n'
                        ]
        config_strs += [f'train_dataset=dict(\n'
                        f'      dataset=dict(\n'
                        f'          data_root   = "{self.params.dataset.dataset_path}",\n'
                        f'          ann_file = "{self.train_ann_file}",\n'
                        f'          data_prefix = dict(img="train/"),\n'
                        f'          metainfo=dict(classes=classes),\n'
                        f'      ) \n'
                        f'  ) \n'
                        ]
        config_strs += [f'test_dataloader=dict(\n'
                        f'  dataset=dict(\n'
                        f'      data_root   = "{self.params.dataset.dataset_path}",\n'
                        f'      ann_file = "{self.val_ann_file}",\n'
                        f'      data_prefix = dict(img="val/"),\n'
                        f'          metainfo=dict(classes=classes),\n'
                        f'    ) \n'
                        f') \n'
                        ]
        config_strs += [f'val_dataloader=dict(\n'
                        f'  dataset=dict(\n'
                        f'      data_root   = "{self.params.dataset.dataset_path}",\n'
                        f'      ann_file = "{self.val_ann_file}",\n'
                        f'      data_prefix = dict(img="val/"),\n'
                        f'      metainfo=dict(classes=classes),\n'
                        f'    ) \n'
                        f') \n'
                        ]
        config_strs += [f'test_evaluator = dict( \n'
                        f'    ann_file="{self.val_ann_file}") \n'
                        ]
        config_strs += [f'val_evaluator = dict( \n'
                        f'    ann_file="{self.val_ann_file}") \n'
                        ]
        config_strs += [f'train_cfg = dict(max_epochs={self.params.training.training_epochs}, type="EpochBasedTrainLoop", val_interval=1)\n']
        # yolox_lr_config_str = \
        #                 f'    num_last_epochs={self.params.training.num_last_epochs},\n' if \
        #                         self.params.training.model_architecture == 'yolox' else ''
        # config_strs += [f'lr_config = dict(\n'
        #                 f'    warmup_by_epoch=True,\n',
        #                 f'    warmup_iters={self.params.training.warmup_epochs},\n',
        #                 f'{yolox_lr_config_str}',
        #                 f')\n',
        #                 ]
        config_strs += [f'load_from   = "{self.params.training.pretrained_checkpoint_path}"']

        # write the config file
        with open(config_file, 'w') as config_fp:
            config_fp.write('\n'.join(config_strs))
        #

        # invoke the distributed training
        if self.params.training.distributed and self.params.training.num_gpus > 0:
            # launcher for the training
            run_launcher = distributed_run.__file__
            run_script = os.path.join(edgeai_mmdetection_tools_path,'train.py')
            argv = [f'--nproc_per_node={self.params.training.num_gpus}',
                    f'--nnodes=1',
                    f'--master_port={self.params.training.training_master_port}',
                    train_module_path,
                    f'--launcher=pytorch',
                    config_file
                    ]
            run_args = [str(arg) for arg in argv]
            run_command = ['python3', run_launcher, run_script] + run_args
        else:
            # Non-cuda mode is currently supported only with non-distributed training
            # os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
            run_script = os.path.join(edgeai_mmdetection_tools_path,'train.py')
            argv = [f'{config_file}']
            run_args = [str(arg) for arg in argv]
            run_command = ['python3', run_script] + run_args
        #
        with open(self.params.training.log_file_path, 'a') as log_fp:
            proc = subprocess.Popen(run_command, stdout=log_fp, stderr=log_fp)
        #
        return proc

    def stop(self):
        if self.quit_event is not None:
            self.quit_event.set()
            return True
        #
        return False

    def get_params(self):
        return self.params
