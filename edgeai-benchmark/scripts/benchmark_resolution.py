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
import sys
import argparse
import functools
import warnings
import copy
import warnings
from edgeai_benchmark import *

try:
    import onnx
except:
    #warnings.warn('onnx could not be imported - this is not required for inference, but may be required for import')
    pass

# the cwd must be the root of the respository
if os.path.split(os.getcwd())[-1] == 'scripts':
    os.chdir('../')
#


if __name__ == '__main__':
    print(f'argv={sys.argv}')
    # the cwd must be the root of the respository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #
    model_selection_default = [
                       'cl-6060',
                       'cl-6090',
                       'cl-6158',
                       'cl-6100',
                       'cl-6110',
                       'cl-6160',
                       'cl-6170',
                       'cl-6180'
                      ]

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('settings_file', type=str, default=None)
    parser.add_argument('--configs_path', type=str)
    parser.add_argument('--models_path', type=str)
    parser.add_argument('--task_selection', type=str, nargs='*')
    parser.add_argument('--model_selection', default=model_selection_default, type=str, nargs='*')
    parser.add_argument('--session_type_dict', type=str, nargs='*')
    parser.add_argument('--input_sizes', default=[512, 1024], type=int, nargs='*')
    cmds = parser.parse_args()

    kwargs = vars(cmds)
    input_sizes = kwargs.pop('input_sizes')
    if 'session_type_dict' in kwargs:
        kwargs['session_type_dict'] = utils.str_to_dict(kwargs['session_type_dict'])
    #
    # these artifacts are meant for only performance measurement - just do a quick import with simple calibration
    # also set the high_resolution_optimization flag for improved performance at high resolution
    runtime_options = {'accuracy_level': 0, 'advanced_options:high_resolution_optimization': 1}

    # the transformations that needs to be applied to the model itself. Note: this is different from pre-processing transforms
    model_transformation_dict = {'input_sizes': input_sizes}

    settings = config_settings.ConfigSettings(cmds.settings_file,
        num_frames=100, calibration_iterations=1, runtime_options=runtime_options,
        model_transformation_dict=model_transformation_dict, **kwargs)

    work_dir = os.path.join(settings.modelartifacts_path, f'{settings.tensor_bits}bits')
    print(f'work_dir: {work_dir}')

    # run the accuracy pipeline
    interfaces.run_accuracy(settings, work_dir)
