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


def get_paretto_front(xy_list, x_index=0, y_index=1, inverse_relaionship=True):
    xy_list = sorted(xy_list, key=lambda x:x[x_index], reverse=inverse_relaionship)
    paretto_front = [xy_list[0]]
    for xy in xy_list[1:]:
        if xy[y_index] >= paretto_front[-1][y_index]:
            paretto_front.append(xy)
        #
    #
    # sort based on first index
    paretto_front = sorted(paretto_front, key=lambda x:x[x_index])
    return paretto_front


def set_model_selection_factor(model_descriptions):
    for m in model_descriptions.values():
        for target_device in m.training.target_devices.keys():
            m.training.target_devices[target_device].model_selection_factor = None
        #
    #

    task_types = set([m.common.task_type for m in model_descriptions.values()])
    target_devices = [list(m.training.target_devices.keys()) for m in model_descriptions.values()]
    target_devices = set([t for t_list in target_devices for t in t_list])
    for target_device in target_devices:
        for task_type in task_types:
            model_desc_list = [m for m in model_descriptions.values() if m.common.task_type == task_type]
            model_desc_list = [m for m in model_desc_list if target_device in list(m.training.target_devices.keys())]
            model_desc_list = [m for m in model_desc_list \
                                   if m.training.target_devices[target_device].performance_fps is not None and
                                      m.training.target_devices[target_device].accuracy_factor is not None]
            performance_fps = [m.training.target_devices[target_device].performance_fps for m in model_desc_list]
            accuracy_factor = [m.training.target_devices[target_device].accuracy_factor for m in model_desc_list]
            xy_list = [(performance_fps[i], accuracy_factor[i], i) for i in range(len(performance_fps))]
            xy_list = get_paretto_front(xy_list)
            for paretto_id, xy in enumerate(xy_list):
                xy_id = xy[2]
                m = model_desc_list[xy_id]
                m.training.target_devices[target_device].model_selection_factor = paretto_id
            #
        #
    #


def get_model_descriptions(params):
    # populate a good pretrained model for the given task
    model_descriptions = training.get_model_descriptions(task_type=params.common.task_type,
                                                         target_device=params.common.target_device,
                                                         training_device=params.training.training_device)

    #
    model_descriptions = utils.ConfigDict(model_descriptions)
    set_model_selection_factor(model_descriptions)
    return model_descriptions


def get_model_description(model_name):
    assert model_name, 'model_name must be specified for get_model_description().' \
        'if model_name is not known, use the method get_model_descriptions() that returns supported models.'
    model_description = training.get_model_description(model_name)
    return model_description


def set_model_description(params, model_description):
    assert model_description is not None, f'could not find pretrained model for {params.training.model_name}'
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
