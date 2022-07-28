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


def init_params(*args, **kwargs):
    default_params = dict(
        common=dict(
            verbose_mode=True,
            projects_path='./data/projects',
            project_path=None,
            project_run_path=None,
            task_type=None,
            target_machine='j7',
            target_device=None,
            # run_name can be any string
            # if {date-time} is given in run_name it will be considered special.
            # will be replaced with datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name='{date-time}',
        ),
        dataset=dict(
            enable=True,
            dataset_name=None,
            dataset_path=None, # dataset split will be created here
            download_path=None,
            extract_path=None,
            split_factor=0.75,
            split_names=('train', 'val'),
            max_num_files=None,
            input_data_path=None, # input images
            input_annotation_path=None, # annotation file
            data_path_splits=None,
            data_dir='images',
            annotation_path_splits=None,
            annotation_dir='annotations',
            annotation_prefix='instances', # change this if your dataset has a different annotation prefix
            dataset_download=True,
            dataset_reload=False
        ),
        training=dict(
            enable=True,
            model_key=None,
            training_backend=None,
            model_name=None,
            model_id=None,
            pretrained_checkpoint_path=None,
            target_devices={},
            project_path=None,
            dataset_path=None,
            training_path=None,
            log_file_path=None,
            model_checkpoint_path=None,
            model_export_path=None,
            model_proto_path=None,
            training_epochs=30,
            warmup_epochs=1,
            num_last_epochs=5,
            batch_size=8,
            learning_rate=2e-3,
            num_classes=None,
            weight_decay=1e-4,
            input_resize=(512 ,512),
            input_cropsize=(512 ,512),
            training_device=None,  # 'cpu', 'cuda'
            num_gpus=0,  # 0,1..4
            distributed=True,
            training_master_port=29500
        ),
        compilation=dict(
            enable=True,
            log_file_path=None,
            compilation_path=None,
            model_compiled_path=None,
            model_packaged_path=None,
            accuracy_level=1,
            tensor_bits=8,
            calibration_frames=50,
            calibration_iterations=50,
            num_frames=None, # inference frame for accuracy test example: 100
            detection_thr=0.3, # threshold for detection: 0.3 for best performance(fps), 0.05 for best accuracy
            save_output=True # save inference outputs
        ),
    )
    params = utils.ConfigDict(default_params, *args, **kwargs)
    return params
