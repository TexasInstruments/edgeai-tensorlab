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


edgeai_modelzoo_path = os.path.abspath('../edgeai-modelzoo')
www_modelzoo_path = 'https://software-dl.ti.com/jacinto7/esd/modelzoo/latest'


_model_descriptions = {
    'ssd_mobilenetv2_fpn_lite_mmdet': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        training=dict(
            training_backend='edgeai_mmdetection',
            model_id='od-8030',
            model_name='ssd_mobilenetv2_fpn_lite_mmdet',
            model_key='ssd_mobilenet_fpn_lite',
            model_architecture='ssd',
            input_resize=(512,512),
            input_cropsize=(512,512),
            pretrained_checkpoint_path=f'{www_modelzoo_path}/models/vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_fpn_lite_512x512_20201110_checkpoint.pth',
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=172, accuracy_factor=27.2)
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': None
            },
            metric=dict(label_offset_pred=0)
        )
    ),
    'ssd_regnetx_800mf_fpn_bgr_lite_mmdet': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        training=dict(
            training_backend='edgeai_mmdetection',
            model_id='od-8050',
            model_name='ssd_regnetx_800mf_fpn_bgr_lite_mmdet',
            model_key='ssd_regnetx_800mf_fpn_bgr_lite',
            model_architecture='ssd',
            input_resize=(512,512),
            input_cropsize=(512,512),
            pretrained_checkpoint_path=f'{www_modelzoo_path}/models/vision/detection/coco/edgeai-mmdet/ssd_regnetx-800mf_fpn_bgr_lite_512x512_20200919_checkpoint.pth',
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=92, accuracy_factor=32.8)
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': None
            },
            metric=dict(label_offset_pred=0)
        )
    ),
    'ssd_regnetx_1p6gf_fpn_bgr_lite_mmdet': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        training=dict(
            training_backend='edgeai_mmdetection',
            model_id='od-8060',
            model_name='ssd_regnetx_1p6gf_fpn_bgr_lite_mmdet',
            model_key='ssd_regnetx_1p6gf_fpn_bgr_lite',
            model_architecture='ssd',
            input_resize=(768,768),
            input_cropsize=(768,768),
            pretrained_checkpoint_path=f'{www_modelzoo_path}/models/vision/detection/coco/edgeai-mmdet/ssd_regnetx-1.6gf_fpn_bgr_lite_768x768_20200923_checkpoint.pth',
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=45, accuracy_factor=37.0)
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': None
            },
            metric=dict(label_offset_pred=0)
        )
    ),
    'yolox_nano_lite_mmdet': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        training=dict(
            training_backend='edgeai_mmdetection',
            model_id='od-8200',
            model_name='yolox_nano_lite_mmdet',
            model_key='yolox_nano_lite',
            model_architecture='yolox',
            input_resize=416,
            input_cropsize=416,
            pretrained_checkpoint_path=f'{www_modelzoo_path}/models/vision/detection/coco/edgeai-mmdet/yolox_nano_lite_416x416_20220214_checkpoint.pth',
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=386, accuracy_factor=24.8)
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': '1213, 1212, 1211, 1197, 1196, 1195, 1181, 1180, 1179'
            },
            metric=dict(label_offset_pred=0)
        )
    ),
    'yolox_tiny_lite_mmdet': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        training=dict(
            training_backend='edgeai_mmdetection',
            model_id='od-8210',
            model_name='yolox_tiny_lite_mmdet',
            model_key='yolox_tiny_lite',
            model_architecture='yolox',
            input_resize=416,
            input_cropsize=416,
            pretrained_checkpoint_path=f'{www_modelzoo_path}/models/vision/detection/coco/edgeai-mmdet/yolox_tiny_lite_416x416_20220217_checkpoint.pth',
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=264, accuracy_factor=30.5)
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': '1213, 1212, 1211, 1197, 1196, 1195, 1181, 1180, 1179'
            },
            metric=dict(label_offset_pred=0)
        )
    ),
    'yolox_s_lite_mmdet': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        training=dict(
            training_backend='edgeai_mmdetection',
            model_id='od-8220',
            model_name='yolox_s_lite_mmdet',
            model_key='yolox_s_lite',
            model_architecture='yolox',
            input_resize=640,
            input_cropsize=640,
            pretrained_checkpoint_path=f'{www_modelzoo_path}/models/vision/detection/coco/edgeai-mmdet/yolox_s_lite_640x640_20220221_checkpoint.pth',
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=107, accuracy_factor=38.3)
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            }
        ),
        compilation=dict(
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': '1213, 1212, 1211, 1197, 1196, 1195, 1181, 1180, 1179'
            },
            metric=dict(label_offset_pred=0)
        )
    ),
    'yolox_m_lite_mmdet': dict(
        common=dict(
            task_type=constants.TASK_TYPE_DETECTION,
        ),
        training=dict(
            training_backend='edgeai_mmdetection',
            model_id='od-8230',
            model_name='yolox_m_lite_mmdet',
            model_key='yolox_m_lite',
            model_architecture='yolox',
            input_resize=640,
            input_cropsize=640,
            pretrained_checkpoint_path=f'{www_modelzoo_path}/models/vision/detection/coco/edgeai-mmdet/yolox_m_lite_20220228_checkpoint.pth',
            target_devices={
                constants.TARGET_DEVICE_TDA4VM: dict(performance_fps=46, accuracy_factor=44.4)
            },
            training_devices={
                constants.TRAINING_DEVICE_CPU: True,
                constants.TRAINING_DEVICE_CUDA: True,
            },
        ),
        compilation=dict(
            runtime_options={
                'advanced_options:output_feature_16bit_names_list': '1546, 1547, 1548, 1562, 1563, 1564, 1578, 1579, 1580'
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
        # training params
        mmdet_path = os.path.abspath('../edgeai-mmdetection')
        dataset_style = 'coco' #'voc' #'coco'
        input_size = self.params.training.input_cropsize if isinstance(self.params.training.input_cropsize, (list,tuple)) else \
            (self.params.training.input_cropsize,self.params.training.input_cropsize)
        base_config_path = os.path.join(mmdet_path, 'configs', 'edgeailite', self.params.training.model_architecture, self.params.training.model_key)

        config_file = os.path.join(self.params.training.training_path, f'{self.params.training.model_name}.py')
        config_strs = []
        config_strs += [f'_base_   = ['
                        f'"{base_config_path}.py"]\n']
        config_strs += [f'work_dir   = "{self.params.training.training_path}"']
        config_strs += [f'data_root   = "{self.params.dataset.dataset_path}"']
        config_strs += [f'total_epochs   = {self.params.training.training_epochs}']
        config_strs += [f'export_model   = True']
        config_strs += [f'optimizer=dict(\n'
                        f'  lr={self.params.training.learning_rate}\n'                           
                        f') \n']
        config_strs += [f'model=dict(\n'
                        f'  bbox_head=dict(\n'
                        f'    num_classes={self.params.training.num_classes}\n'
                        f'  ) \n'                            
                        f') \n']
        config_strs += [f'data=dict(\n'
                        f'  samples_per_gpu={self.params.training.batch_size},\n'                        
                        f'  train=dict(\n'
                        f'    dataset=dict(\n'
                        f'      type="ModelMakerDataset",\n'
                        f'      ann_file = "{self.train_ann_file}",\n'
                        f'      img_prefix = "{self.params.dataset.dataset_path}/train",\n'
                        f'      classes = {self.object_categories}\n'
                        f'    ) \n'                            
                        f'  ), \n'                        
                        f'  val=dict(\n'
                        f'    type="ModelMakerDataset",\n'                        
                        f'    ann_file = "{self.val_ann_file}",\n'
                        f'    img_prefix = "{self.params.dataset.dataset_path}/val",\n'
                        f'    classes = {self.object_categories}\n'                        
                        f'  ) \n'
                        f')\n'
                        ]
        config_strs += [f'runner = dict(\n'
                        f'    type="EpochBasedRunner", max_epochs={self.params.training.training_epochs}\n'
                        f')\n',
                        ]
        yolox_lr_config_str = \
                        f'    num_last_epochs={self.params.training.num_last_epochs},\n' if \
                                self.params.training.model_architecture == 'yolox' else ''
        config_strs += [f'lr_config = dict(\n'
                        f'    warmup_by_epoch=True,\n',
                        f'    warmup_iters={self.params.training.warmup_epochs},\n',
                        f'{yolox_lr_config_str}',
                        f')\n',
                        ]
        config_strs += [f'load_from   = "{self.params.training.pretrained_checkpoint_path}"']

        # write the config file
        with open(config_file, 'w') as config_fp:
            config_fp.write('\n'.join(config_strs))
        #

        # invoke the distributed training
        train_module_path = f'{mmdet_path}/tools/train.py'
        if self.params.training.distributed and self.params.training.num_gpus > 0:
            sys.argv = [sys.argv[0],
                        f'--nproc_per_node={self.params.training.num_gpus}',
                        f'--nnodes=1',
                        f'--master_port={self.params.training.training_master_port}',
                        train_module_path,
                        f'--launcher=pytorch',
                        config_file
                        ]
            # launch the training
            distributed_launch.main()
        else:
            # Non-cuda mode is currently supported only with non-distributed training
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
            train_module = utils.import_file_or_folder(train_module_path)
            # sys.argv = [sys.argv[0], f'--gpus={self.params.training.num_gpus}', '--no-validate', f'{config_file}']
            sys.argv = [sys.argv[0], f'{config_file}']
            args = train_module.parse_args()
            train_module.main(args)
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
