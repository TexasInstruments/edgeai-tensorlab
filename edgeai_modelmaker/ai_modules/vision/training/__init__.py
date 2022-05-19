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
from . import edgeai_torchvision
from . import edgeai_mmdetection


# list all the modules here to add pretrained models
_supported_models = {}

# detection models
_supported_models.update(edgeai_mmdetection.detection.get_supported_models())
# the detection models in edgeai-mmdetection are superior to edgeai-torchvision, so commenting out these for now.
#_supported_models.update(edgeai_torchvision.detection.get_supported_models())

# classification models
_supported_models.update(edgeai_torchvision.classification.get_supported_models())


def get_supported_models(task_type=None, target_device=None, training_device=None):
    supported_models_selected = copy.deepcopy(_supported_models)
    if task_type is not None:
        supported_models_selected = {k:v for k, v in supported_models_selected.items() if v['common']['task_type'] == task_type}
    #
    if target_device is not None:
        supported_models_selected = {k:v for k, v in supported_models_selected.items() if target_device in v['training']['target_devices']}
    #
    if training_device is not None:
        supported_models_selected = {k:v for k, v in supported_models_selected.items() if training_device in \
                                      v['training']['training_devices'] and v['training']['training_devices']}
    #
    return supported_models_selected


def get_supported_model(model_key):
    supported_models = get_supported_models()
    return supported_models[model_key] if model_key in supported_models else None


def get_target_module(backend_name, task_type):
    this_module = sys.modules[__name__]
    backend_package = getattr(this_module, backend_name)
    target_module = getattr(backend_package, task_type)
    return target_module
