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

import datetime
from ... import utils
from . import constants


def init_params(*args, **kwargs):
    default_params = dict(
        common=dict(
            verbose_mode=True,
            download_path='./data/downloads',
            projects_path='./data/projects',
            project_path=None,
            project_run_path=None,
            task_type=None,
            # does not affect functionality - but setting to pc avoids (wrong) inference time capture during compilation
            target_machine='pc', #'evm',
            target_device=None,
            # run_name can be any string, but there are some special cases:
            # {date-time} will be replaced with datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # {model_name} will be replaced with the name of the model
            run_name='{date-time}/{model_name}',
        ),
        download=None,
        dataset=dict(
            enable=True,
            dataset_name=None,
            dataset_path=None, # dataset split will be created here
            extract_path=None,
            split_factor=0.80,
            split_names=('train', 'val'),
            max_num_files=10000,
            input_data_path=None, # input images
            input_annotation_path=None, # annotation file
            data_path_splits=None,
            data_dir='images',
            annotation_path_splits=None,
            annotation_dir='annotations',
            annotation_prefix='instances', # change this if your dataset has a different annotation prefix
            annotation_format='coco_json',
            dataset_download=True,
            dataset_reload=False
        ),
        training=dict(
            enable=True,
            model_name=None,
            model_training_id=None,
            training_backend=None,
            pretrained_checkpoint_path=None,
            pretrained_weight_state_dict_name=None,
            target_devices={},
            project_path=None,
            dataset_path=None,
            training_path=None,
            log_file_path=None,
            log_summary_regex=None,
            summary_file_path=None,
            model_checkpoint_path=None,
            model_export_path=None,
            model_proto_path=None,
            model_packaged_path=None,
            training_epochs=15,
            warmup_epochs=1,
            num_last_epochs=5,
            batch_size=8,
            learning_rate=1e-3,
            num_classes=None,
            weight_decay=1e-4,
            input_resize=(512 ,512),
            input_cropsize=(512 ,512),
            training_device=None,  # 'cpu', 'cuda'
            num_gpus=0,  # 0,1..4
            distributed=True,
            training_master_port=29500,
            with_background_class=None,
            train_output_path=None,
            properties=[
                dict(type="group", dynamic=False, name="train_group", label="Training Parameters",
                     default=["training_epochs", "learning_rate", "batch_size", "weight_decay"]),
                dict(label="Epochs", name="training_epochs", type="integer", default=50, min=1, max=300),
                dict(label="Learning Rate", name="learning_rate", type="float", default=0.001, min=0.0001, max=0.1,
                     decimal_places=3, increment=0.001),
                dict(label="Batch Size", name="batch_size", type="integer", default=8, min=1, max=128),
                dict(label="Weight Decay", name="weight_decay", type="float", default=0.0001, min=0.0001, max=0.1,)
                ]
        ),
        compilation=dict(
            enable=True,
            compile_preset_name=None,
            model_compilation_id=None,
            compilation_path=None, # top level compilation path
            model_compiled_path=None, # compiled path for the model
            log_file_path=None,
            log_summary_regex=None,
            summary_file_path=None,
            output_tensors_path=None,
            model_packaged_path=None,
            model_visualization_path=None,
            # accuracy_level=1,
            tensor_bits=8,
            calibration_frames=10,
            calibration_iterations=10,
            num_frames=None, # inference frames for accuracy measurement. example: 100. None means entire validation set.
            num_output_frames=50, # inference frames for example output. example: 50
            detection_threshold=0.3, # threshold for detection: 0.6 for visualization, 0.3 atleast for good performance(fps), 0.05 for best accuracy
            detection_top_k=200, # num boxes to preserve in nms: 200 for best performance(fps), 500 for best accuracy.
            save_output=True, # save inference outputs
            tidl_offload=True,
            input_optimization=True, # if this is set, the compilation tool will try to fold mean and scale inside the model.
            log_file=True, # capture logs into log_file
            compile_output_path=None,
            properties=[dict(
                label="Compilation Preset", name="compile_preset_name", type="enum",
                default='default_preset',
                enum=[
                    # {"value": constants.COMPILATION_FORCED_SOFT_NPU, "label": "Forced Software NPU", "tooltip": "Only for F28P55, to disable HW NPU"},
                      {"value": 'best_accuracy_preset', "label": "best accuracy preset", "tooltip": "Best Accuracy Inference Mode"},
                      {"value": 'high_accuracy_preset', "label": "high accuracy preset", "tooltip": "high Accuracy Inference Mode"},
                      {"value": 'default_preset', "label": "default preset", "tooltip": "Default Inference Mode"},
                      {"value": 'high_speed_preset', "label": "high speed preset", "tooltip": "high Speed Inference Mode"},
                      {"value": 'best_speed_preset', "label": "best speed preset", "tooltip": "best Speed Inference Mode"},
                      ])
            ],
        ),
    )
    params = utils.ConfigDict(default_params, *args, **kwargs)
    return params
