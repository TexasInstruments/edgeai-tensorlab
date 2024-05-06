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
from torch.distributed import launch as distributed_launch

from ... import constants
from ..... import utils

this_dir_path = os.path.dirname(os.path.abspath(__file__))
repo_parent_path = os.path.abspath(os.path.join(this_dir_path, '../../../../../../'))

edgeai_modelzoo_path = os.path.join(repo_parent_path, 'edgeai-modelzoo')
www_modelzoo_path = 'https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01'
edgeai_yolox_path = os.path.join(repo_parent_path, 'edgeai-mmpose', 'projects', 'yolox-pose')
edgeai_yoloxpose_tools_path = os.path.join(edgeai_yolox_path, 'tools')

# TODO: Need to change model urls with yolox_pose models
model_urls = {
    'yolox-pose_tiny_4xb64-300e_coco': [
        {
            'download_url': f'{www_modelzoo_path}/models/vision/detection/coco/edgeai-mmdet/yolox_femto_lite_320x320_20230407_checkpoint.pth',
            'download_path': os.path.join('{download_path}', 'pretrained', 'yolox-pose_tiny_4xb64-300e_coco')
        },
    ],
    'yolox-pose_s_8xb32-300e_coco_lite': [
        {
            'download_url': f'/home/a0504871/work/ti/mmpose/projects/yolox-pose/work_dirs/yolox-pose_s_8xb32-300e_coco_lite/best_coco_AP_epoch_397.pth',
            'download_path': os.path.join('{download_path}', 'pretrained', 'yolox-pose_s_8xb32-300e_coco_lite')
        },
    ]
}

# TODO: Need to change model descriptions according to yolox_pose models
_model_descriptions = {
    'yolox-pose_tiny_4xb64-300e_coco': dict(
        common=dict(
            task_type=constants.TASK_TYPE_KEYPOINT_DETECTION,
        ),
        download=model_urls['yolox-pose_tiny_4xb64-300e_coco'],
        training=dict(
            training_backend='edgeai_mmpose',
            model_name='yolox-pose_tiny_4xb64-300e_coco',
            model_training_id='yolox-pose_tiny_4xb64-300e_coco',
            model_architecture='yolox',
            input_resize=416,
            input_cropsize=416,
            pretrained_checkpoint_path=model_urls['yolox-pose_tiny_4xb64-300e_coco'][0],
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_KEYPOINT_DETECTION],
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=None, performance_infer_time_ms=4.92,
                                                     accuracy_factor=47.4, accuracy_unit='AP50%', accuracy_factor2=30.5, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM62A: dict(performance_fps=None, performance_infer_time_ms=15.32,
                                                     accuracy_factor=47.4, accuracy_unit='AP50%', accuracy_factor2=30.5, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM68A: dict(performance_fps=None, performance_infer_time_ms=4.92,
                                                     accuracy_factor=47.4, accuracy_unit='AP50%', accuracy_factor2=30.5, accuracy_unit2='AP[.5:.95]%')
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='od-8210',
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': '1213, 1212, 1211, 1197, 1196, 1195, 1181, 1180, 1179'
            },
            metric=dict(label_offset_pred=0)
        )
    ),
    'yolox-pose_s_8xb32-300e_coco_lite': dict(
        common=dict(
            task_type=constants.TASK_TYPE_KEYPOINT_DETECTION,
        ),
        download=model_urls['yolox-pose_s_8xb32-300e_coco_lite'],
        training=dict(
            training_backend='edgeai_mmpose',
            model_name='yolox-pose_s_8xb32-300e_coco_lite',
            model_training_id='yolox-pose_s_8xb32-300e_coco_lite',
            model_architecture='yolox',
            input_resize=640,
            input_cropsize=640,
            pretrained_checkpoint_path=model_urls['yolox-pose_s_8xb32-300e_coco_lite'][0],
            batch_size=constants.TRAINING_BATCH_SIZE_DEFAULT[constants.TASK_TYPE_DETECTION],
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=None, performance_infer_time_ms=10.19,
                                                     accuracy_factor=56.9, accuracy_unit='AP50%', accuracy_factor2=38.3, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM62A: dict(performance_fps=None, performance_infer_time_ms=43.85,
                                                     accuracy_factor=56.9, accuracy_unit='AP50%', accuracy_factor2=38.3, accuracy_unit2='AP[.5:.95]%'),
                constants.TARGET_DEVICE_AM68A: dict(performance_fps=None, performance_infer_time_ms=10.19,
                                                     accuracy_factor=56.9, accuracy_unit='AP50%', accuracy_factor2=38.3, accuracy_unit2='AP[.5:.95]%'), #TODO: this has to be corrected
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            model_compilation_id='od-8220',
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': '1213, 1212, 1211, 1197, 1196, 1195, 1181, 1180, 1179'
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

        # num_classes
        self.train_ann_file = f'{self.params.dataset.dataset_path}/annotations/{self.params.dataset.annotation_prefix}_{self.params.dataset.split_names[0]}.json'
        self.val_ann_file = f'{self.params.dataset.dataset_path}/annotations/{self.params.dataset.annotation_prefix}_{self.params.dataset.split_names[1]}.json'
        with open(self.train_ann_file) as train_ann_fp:
            train_anno = json.load(train_ann_fp)
            categories = train_anno['categories']
            self.num_keypoints = len(categories[0]['keypoints'])
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
        # training params
        dataset_style = 'coco' #'voc' #'coco'
        input_size = self.params.training.input_cropsize if isinstance(self.params.training.input_cropsize, (list,tuple)) else \
            (self.params.training.input_cropsize,self.params.training.input_cropsize)
        base_config_path = os.path.join(edgeai_yolox_path, 'configs', 'edgeailite', self.params.training.model_training_id)

        config_file = os.path.join(self.params.training.training_path, f'{self.params.training.model_name}.py')
        config_strs = []
        base_meta_file = 'configs/_base_/datasets/coco.py'
        max_epochs = self.params.training.training_epochs
        intermediate_epoch = 2
        num_last_epochs = 2


        config_strs += [f'_base_   = ['
                        f'"{base_config_path}.py"]\n']
        config_strs += [f'model=dict(\n'
                        f'  init_cfg=dict(checkpoint="{self.params.training.pretrained_checkpoint_path}"),\n'
                        f'  bbox_head=dict(\n'
                        f'    head_module=dict(num_keypoints={self.params.training.num_keypoints}),\n'
                        f'    loss_pose=dict(metainfo="{base_meta_file}")\n'
                        f'  ), \n'  
                        f'  train_cfg=dict(assigner=dict(oks_calculator=dict(metainfo="{base_meta_file}"))),\n'
                        f') \n']
        config_strs += [f'train_dataloader=dict(\n'
                        f'  dataset=dict(\n'
                        f'      data_root="{self.params.dataset.dataset_path}",\n'
                        f'      ann_file="{self.params.dataset.annotation_path_splits[0]}",\n'
                        f'      data_prefix=dict(img="{self.params.dataset.split_names[0]}")\n'
                        f'  )\n'
                        f') \n']
        config_strs += [f'val_dataloader=dict(\n'
                        f'  dataset=dict(\n'
                        f'      data_root="{self.params.dataset.dataset_path}",\n'
                        f'      ann_file="{self.params.dataset.annotation_path_splits[1]}",\n'
                        f'      data_prefix=dict(img="{self.params.dataset.split_names[1]}")\n'
                        f'  )\n'
                        f') \n']
        config_strs += [f'val_evaluator=dict(ann_file="{self.params.dataset.annotation_path_splits[1]}")\n']
        config_strs += [f'optim_wrapper=dict(optimizer=dict(lr={self.params.training.learning_rate}))\n']
        config_strs += [f'param_scheduler=[\n'
                        f'dict(\n'
                        f'  type="mmdet.QuadraticWarmupLR",\n'
                        f'  by_epoch=True,\n'
                        f'  begin=0,\n'
                        f'  end={intermediate_epoch},\n'
                        f'  convert_to_iter_based=True),\n'
                        f'dict(\n'
                        f'  type="CosineAnnealingLR",\n'
                        f'  eta_min={self.params.training.learning_rate * 0.05},\n'
                        f'  begin={intermediate_epoch},\n'
                        f'  T_max={max_epochs - num_last_epochs},\n'
                        f'  end={max_epochs - num_last_epochs},\n'
                        f'  by_epoch=True,\n'
                        f'  convert_to_iter_based=True),\n'
                        f'dict(\n'
                        f'  type="ConstantLR",\n'
                        f'  by_epoch=True,\n'
                        f'  factor=1,\n'
                        f'  begin={max_epochs - num_last_epochs},\n'
                        f'  end={max_epochs},)\n'
                        f']\n']
        config_strs += [f'train_cfg=dict(\n'
                        f'  max_epochs={max_epochs},\n'
                        f'  val_interval=10,\n'
                        f'  dynamic_intervals=[({max_epochs - num_last_epochs}, 1)]\n'
                        f')\n']

        # write the config file
        with open(config_file, 'w') as config_fp:
            config_fp.write('\n'.join(config_strs))
        #

        # invoke the distributed training
        if self.params.training.distributed and self.params.training.num_gpus > 0:
            train_module_path = f'{edgeai_yolox_path}/tools/train.py'
            sys.argv = [sys.argv[0],
                        f'--nproc_per_node={self.params.training.num_gpus}',
                        f'--nnodes=1',
                        f'--master_port={self.params.training.training_master_port}',
                        train_module_path,
                        f'--launcher=pytorch',
                        config_file
                        ]

            sys.path.insert(0, edgeai_yolox_path)
            # launch the training
            distributed_launch.main()
        else:
            # Non-cuda mode is currently supported only with non-distributed training
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
            # sys.argv = [sys.argv[0], f'--gpus={self.params.training.num_gpus}', '--no-validate', f'{config_file}']
            sys.argv = [sys.argv[0], f'{config_file}']
            sys.path.insert(0, edgeai_yolox_path)
            # import dynamically - force_import every time to avoid clashes with scripts in other repositories
            train_module = utils.import_file_or_folder(os.path.join(edgeai_yoloxpose_tools_path,'train'),
                __name__, force_import=True)
            train_module.main()
        #
        return self.params

    def stop(self):
        if self.quit_event is not None:
            self.quit_event.set()
            return True
        #
        return False

    def get_params(self):
        return self.params
