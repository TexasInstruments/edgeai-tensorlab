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


def _get_paretto_front_best(xy_list, x_index=0, y_index=1, inverse_relaionship=True):
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


def _get_paretto_front_approx(xy_list, x_index=0, y_index=1, inverse_relaionship=True):
    # normalize the values
    min_x = min(xy[0] for xy in xy_list)
    max_x = max(xy[0] for xy in xy_list)
    min_y = min(xy[1] for xy in xy_list)
    max_y = max(xy[1] for xy in xy_list)
    norm_xy_list = [[(xy[0]-min_x+1)/(max_x-min_x+1), (xy[1]-min_y+1)/(max_y-min_y+1), xy[2]] for xy in xy_list]
    if inverse_relaionship:
        efficiency_list = [list(xy)+[xy[y_index]*xy[x_index]] for xy in norm_xy_list]
    else:
        efficiency_list = [list(xy)+[xy[y_index]/xy[x_index]] for xy in norm_xy_list]
    #
    efficiency_list = sorted(efficiency_list, key=lambda x:x[-1], reverse=True)
    # take the good models
    num_models_selected = max(len(efficiency_list)*2//3, 1)
    efficiency_list = efficiency_list[:num_models_selected]
    selected_indices = [xy[2] for xy in efficiency_list]
    selected_entries = [xy for xy in xy_list if xy[2] in selected_indices]
    # sort based on first index
    paretto_front = sorted(selected_entries, key=lambda x:x[x_index])
    return paretto_front


def get_paretto_front_combined(xy_list, x_index=0, y_index=1, inverse_relaionship=True):
    paretto_front_best = _get_paretto_front_best(xy_list, x_index=x_index, y_index=y_index, inverse_relaionship=inverse_relaionship)
    paretto_front_approx = _get_paretto_front_approx(xy_list, x_index=x_index, y_index=y_index, inverse_relaionship=inverse_relaionship)
    paretto_front_combined = paretto_front_best + paretto_front_approx
    # de-duplicate
    selected_indices = [xy[2] for xy in paretto_front_combined]
    selected_indices = set(selected_indices)
    paretto_front = [xy for xy in xy_list if xy[2] in selected_indices]
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
            xy_list = get_paretto_front_combined(xy_list)
            for paretto_id, xy in enumerate(xy_list):
                xy_id = xy[2]
                m = model_desc_list[xy_id]
                m.training.target_devices[target_device].model_selection_factor = paretto_id
            #
        #
    #


def get_training_module_descriptions(params):
    # populate a good pretrained model for the given task
    training_module_descriptions = training.get_training_module_descriptions(target_device=params.common.target_device,
                                                         training_device=params.training.training_device)
    #
    training_module_descriptions = utils.ConfigDict(training_module_descriptions)
    return training_module_descriptions


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
    return constants.PRESET_DESCRIPTIONS


def get_target_device_descriptions(params):
    return constants.TARGET_DEVICE_DESCRIPTIONS


def get_sample_dataset_descriptions(params):
    return constants.SAMPLE_DATASET_DESCRIPTIONS


def get_task_descriptions(params):
    return constants.TASK_DESCRIPTIONS


def get_help_descriptions(params):
    return {
        'common': {
        },
        'dataset': {
        },
        'training': {
            'training_epochs': {
                'name': 'Epochs',
                'description': 'Epoch is a term that is used to indicate a pass over the entire training dataset. '
                           'It is a hyper parameter that can be tuned to get best accuracy. '
                           'Eg. A model trained for 30 Epochs may give better accuracy than a model trained for 15 Epochs.'
            },
            'learning_rate': {
                'name': 'Learning Rate',
                'description': 'Learning Rate determines the step size used by the optimization algorithm '
                               'at each iteration while moving towards the optimal solution. '
                               'It is a hyper parameter that can be tuned to get best accuracy. '
                               'Eg. A small Learning Rate typically gives good accuracy while fine tuning a model for a different task.'
            },
            'batch_size': {
                'name': 'Batch Size',
                'description': 'Batch size specifies the number of inputs that are propagated through the '
                               'neural network in one iteration. Several such iterations make up one Epoch.'
                               'Higher batch size require higher memory and too low batch size can '
                               'typically impact the accuracy.'
            },
            'weight_decay': {
                'name': 'Weight Decay',
                'description': 'Weight decay is a regularization technique that can improve '
                               'stability and generalization of a machine learning algorithm. '
                               'It is typically done using L2 regularization that penalizes parameters '
                               '(weights, biases) according to their L2 norm.'
            },
        },
        'compilation': {
            'calibration_frames': {
                'name': 'Calibration Frames',
                'description': 'Calibration is a process of improving the accuracy during fixed point quantization. '
                               'Typically, higher number of Calibration Frames give higher accuracy, but it can also be time consuming.'
            },
            'calibration_iterations': {
                'name': 'Calibration Iterations',
                'description': 'Calibration is a process of improving the accuracy during fixed point quantization. Calibration happens in iterations. '
                               'Typically, higher number of Calibration Iterations give higher accuracy, but it can also be time consuming.'
            },
            'tensor_bits': {
                'name': 'Tensor Bits',
                'description': 'Bitdepth used to quantize the weights and activations in the neural network. '
                               'The neural network inference happens at this bit precision. '
            },
            'detection_threshold': {
                'name': 'Detection Threshold',
                'description': 'Also called Confidence Threshold. A threshold used to select good detection boxes. '
                               'This is typically applied before a before the Non Max Suppression. '
                               'Higher Detection Threshold means less false detections (False Positives), '
                               'but may also result in misses (False Negatives). '
            },
            'detection_top_k': {
                'name': 'Detection TopK',
                'description': 'Number of detection boxes to be selected during the initial shortlisting before the Non Max Suppression.'
                               'A higher number is typically used while measuring accuracy, but may impact the performance. '
            }
        }
    }
