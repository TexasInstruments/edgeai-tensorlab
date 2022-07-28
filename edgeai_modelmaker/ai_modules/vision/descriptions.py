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


from . import constants
from ... import utils
from . import datasets
from . import training
from . import compilation
from .params import init_params


def get_model_descriptions(params):
    # populate a good pretrained model for the given task
    model_descriptions = training.get_model_descriptions(task_type=params.common.task_type,
                                                         target_device=params.common.target_device,
                                                         training_device=params.training.training_device)
    return model_descriptions


def get_model_description(model_key):
    assert model_key, 'model_key must be specified for get_model_description().' \
        'if model_key is not known, use the method get_model_descriptions() that returns supported models.'
    model_description = training.get_model_description(model_key)
    return model_description


def set_model_description(params, model_description):
    assert model_description is not None, f'could not find pretrained model for {params.training.model_key}'
    assert params.common.task_type == model_description['common']['task_type'], \
        f'task_type: {params.common.task_type} does not match the pretrained model'
    # get pretrained model checkpoint and other details
    params.update(model_description)
    return params


def get_preset_descriptions(params):
    presets = dict(
        default_preset=None,  # not specified here - use the models values
        high_speed_preset=dict(
            compilation=dict(
                calibration_frames=10,
                calibration_iterations=10,
                detection_thr=0.3
            )
        ),
        high_accuracy_preset=dict(
            compilation=dict(
                calibration_frames=50,
                calibration_iterations=50,
                detection_thr=0.05
            )
        )
    )
    return presets


def get_target_device_descriptions(params):
    return constants.TARGET_DEVICE_DESCRIPTIONS


def get_sample_dataset_descriptions(params):
    return constants.SAMPLE_DATASET_DESCRIPTIONS


def get_task_descriptions(params):
    return constants.TASK_DESCRIPTIONS
