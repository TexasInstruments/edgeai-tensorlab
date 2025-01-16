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
from edgeai_benchmark import *
from edgeai_benchmark.config_utils import str_to_bool

if __name__ == '__main__':
    print(f'argv: {sys.argv}')
    # the cwd must be the root of the respository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('settings_file', type=str, default=None)
    parser.add_argument('--target_device', type=str)
    parser.add_argument('--tensor_bits', type=utils.str_to_int)
    parser.add_argument('--configs_path', type=str)
    parser.add_argument('--models_path', type=str)
    parser.add_argument('--task_selection', type=str, nargs='*')
    parser.add_argument('--runtime_selection', type=str, nargs='*')
    parser.add_argument('--model_selection', type=utils.str_or_none, nargs='*')
    parser.add_argument('--model_shortlist', type=utils.int_or_none)
    parser.add_argument('--session_type_dict', type=str, nargs='*')
    parser.add_argument('--num_frames', type=int)
    parser.add_argument('--calibration_frames', type=int)
    parser.add_argument('--calibration_iterations', type=int)
    parser.add_argument('--run_import', type=utils.str_to_bool)
    parser.add_argument('--run_inference', type=utils.str_to_bool)
    parser.add_argument('--modelartifacts_path', type=str)
    parser.add_argument('--modelpackage_path', type=str)
    parser.add_argument('--dataset_loading', type=utils.str_to_bool, default=False)
    parser.add_argument('--parallel_devices', type=utils.int_or_none)
    parser.add_argument('--parallel_processes', type=int)
    parser.add_argument('--fast_calibration_factor', type=utils.float_or_none)
    parser.add_argument('--experimental_models', type=utils.str_to_bool)
    parser.add_argument('--additional_models', type=utils.str_to_bool)
    parser.add_argument('--enable_logging', type=utils.str_to_bool)
    parser.add_argument('--models_list_file', type=str, default=None)
    parser.add_argument('--get_run_dir', type=str_to_bool, default=False)
    # this is only for compatibility with benchmark_modelzoo.py
    parser.add_argument('--overall_timeout', type=utils.float_or_none)
    parser.add_argument('--instance_timeout', type=utils.float_or_none)
    cmds = parser.parse_args()

    kwargs = vars(cmds)
    if 'session_type_dict' in kwargs:
        kwargs['session_type_dict'] = utils.str_to_dict(kwargs['session_type_dict'])
    #
    settings = config_settings.ConfigSettings(cmds.settings_file, **kwargs)
    print(f'settings: {settings}')
    sys.stdout.flush()

    work_dir = os.path.join(settings.modelartifacts_path, f'{settings.tensor_bits}bits')
    print(f'work_dir: {work_dir}')

    if kwargs['models_list_file'] is None:
        kwargs['models_list_file'] = os.path.join(settings.modelartifacts_path, "models_list.txt")

    # get a dict of model configs
    pipeline_configs = interfaces.get_configs(settings, work_dir)

    # filter the configs
    pipeline_configs = pipelines.PipelineRunner(settings, pipeline_configs).get_pipeline_configs()

    model_keys = pipeline_configs.keys()
    with open(kwargs['models_list_file'], "w") as fp:
        for model_key in model_keys:
            fp.write(f"{model_key}")
            if cmds.get_run_dir:
                run_dir = pipeline_configs[model_key]['session'].kwargs['run_dir']
                fp.write(f" {run_dir}\n")
            #
        #
    #
