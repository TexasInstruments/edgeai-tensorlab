# Copyright (c) 2018-2021, Texas Instruments Incorporated
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
import copy
import sys

from .. import constants


# list all the modules here to add pretrained models
_model_descriptions = {}
_training_module_descriptions = {}


# edgeai-torchvision
from . import edgeai_torchvision

## classification
_model_descriptions.update(edgeai_torchvision.classification.get_model_descriptions())
## segmentation
_model_descriptions.update(edgeai_torchvision.segmentation.get_model_descriptions())
## _training_module_descriptions
_training_module_descriptions.update({'edgeai_torchvision':[constants.TASK_TYPE_CLASSIFICATION, constants.TASK_TYPE_SEGMENTATION]})

# edgeai-mmdetection
from . import edgeai_mmdetection
_model_descriptions.update(edgeai_mmdetection.detection.get_model_descriptions())
_training_module_descriptions.update({'edgeai_mmdetection':[constants.TASK_TYPE_DETECTION]})


# edgeai-mmpose
# if constants.PLUGINS_ENABLE_EXTRA:
    # from . import edgeai_mmpose
    # _model_descriptions.update(edgeai_mmpose.keypoint_detection.get_model_descriptions())
    # _training_module_descriptions.update({'edgeai_mmpose':[constants.TASK_TYPE_KEYPOINT_DETECTION]})
#

# edgeai-yolox
if constants.PLUGINS_ENABLE_EXTRA:
    from . import edgeai_yolox
    _model_descriptions.update(edgeai_yolox.keypoint_detection.get_model_descriptions())
    _training_module_descriptions.update({'edgeai_yolox':[constants.TASK_TYPE_KEYPOINT_DETECTION]})
#


def get_training_module_descriptions(target_device=None, training_device=None):
    return _training_module_descriptions


def get_model_descriptions(task_type=None, target_device=None, training_device=None):
    model_descriptions_selected = copy.deepcopy(_model_descriptions)
    if task_type is not None:
        model_descriptions_selected = {k:v for k, v in model_descriptions_selected.items() if v['common']['task_type'] == task_type}
    #
    if target_device is not None:
        model_descriptions_selected = {k:v for k, v in model_descriptions_selected.items() if target_device in v['training']['target_devices']}
    #
    if training_device is not None:
        model_descriptions_selected = {k:v for k, v in model_descriptions_selected.items() if training_device in \
                                      v['training']['training_devices'] and v['training']['training_devices']}
    #
    return model_descriptions_selected


def get_model_description(model_name):
    model_descriptions = get_model_descriptions()
    return model_descriptions[model_name] if model_name in model_descriptions else None


def get_target_module(backend_name, task_type):
    this_module = sys.modules[__name__]
    try:
        backend_package = getattr(this_module, backend_name)
    except Exception as e:
        print(f"get_target_module(): The requested module could not be found: {backend_name}. {str(e)}")
        return None
    #
    try:
        target_module = getattr(backend_package, task_type)
    except Exception as e:
        print(f"get_target_module(): The task_type {task_type} could not be found in the module {backend_name}. {str(e)}")
        return None
    #
    return target_module
