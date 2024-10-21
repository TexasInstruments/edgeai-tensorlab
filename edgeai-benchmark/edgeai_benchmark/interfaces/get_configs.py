# Copyright (c) 2018-2021, Texas Instruments
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

import os

from .. import datasets
from .get_configs_from_module import *
from .get_configs_from_file import *


def get_configs(settings, work_dir):
    # initialize the dataset place holders.
    if settings.dataset_cache is None:
        settings.dataset_cache = datasets.initialize_datasets(settings)
    #
    # now get the config dictionaries
    is_config_file = (os.path.splitext(settings.configs_path)[-1] == '.yaml')
    if is_config_file:
        print(f'Using model config(s) from file: {settings.configs_path}')
        pipeline_configs = get_configs_from_file(settings, work_dir)
    else:
        print(f'Using model configs from Python module: {settings.configs_path}')
        pipeline_configs = get_configs_from_module(settings, work_dir)
    #
    return pipeline_configs


def select_configs(settings, work_dir, session_name=None, remove_models=False):
    # initialize the dataset place holders.
    if settings.dataset_cache is None:
        settings.dataset_cache = datasets.initialize_datasets(settings)
    #
    # now get the config dictionaries
    is_config_file = (os.path.splitext(settings.configs_path)[-1] == '.yaml')
    if is_config_file:
        print(f'Selecting model config(s) from file: {settings.configs_path}')
        pipeline_configs = select_configs_from_file(settings, work_dir, session_name=session_name, remove_models=remove_models)
    else:
        print(f'Selecting model configs from Python module: {settings.configs_path}')
        pipeline_configs = select_configs_from_module(settings, work_dir, session_name=session_name, remove_models=remove_models)
    #
    return pipeline_configs
