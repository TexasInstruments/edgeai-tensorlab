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
import warnings
import subprocess
import tqdm

from edgeai_benchmark import *


def get_arg_parser():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('settings_file', type=str, default=None)
    parser.add_argument('--target_device', type=str)
    parser.add_argument('--tensor_bits', type=utils.str_to_int)
    parser.add_argument('--configs_path', type=str, help='Path to configs. Can be one of: '
                        '\n  Python module, such as in ./configs OR '
                        '\n  a configs file such as ../edgeai-modelzoo/models/configs.yaml. '
                        '\n  default is Python module at: ./configs')
    parser.add_argument('--models_path', type=str)
    parser.add_argument('--task_selection', type=str, nargs='*')
    parser.add_argument('--runtime_selection', type=str, nargs='*')
    parser.add_argument('--model_selection', type=str, nargs='*')
    parser.add_argument('--model_shortlist', type=utils.int_or_none)
    parser.add_argument('--session_type_dict', type=str, nargs='*')
    parser.add_argument('--num_frames', type=int)
    parser.add_argument('--calibration_frames', type=int)
    parser.add_argument('--calibration_iterations', type=int)
    parser.add_argument('--run_import', type=utils.str_to_bool)
    parser.add_argument('--run_inference', type=utils.str_to_bool)
    parser.add_argument('--modelartifacts_path', type=str)
    parser.add_argument('--modelpackage_path', type=str)
    parser.add_argument('--dataset_loading', type=utils.str_to_bool, default=True)
    parser.add_argument('--parallel_devices', type=utils.int_or_none)
    parser.add_argument('--parallel_processes', type=int)
    parser.add_argument('--fast_calibration_factor', type=utils.float_or_none)
    parser.add_argument('--experimental_models', type=utils.str_to_bool)
    parser.add_argument('--additional_models', type=utils.str_to_bool)
    parser.add_argument('--models_list_file', type=str, default=None)
    return parser


def run_one_model(kwargs, model_selection, run_dir):
    cmd_args = []
    for key, value in kwargs.items():
        if key not in ('settings_file', 'models_list_file'):
            cmd_args += [f'--{key}']
            cmd_args += [f'{value}']
        #
    #

    os.makedirs(run_dir, exist_ok=True)
    log_filename = os.path.join(run_dir, 'run.log')
    with open(log_filename, 'w') as log_fp:
        command = ['python3',  './scripts/benchmark_modelzoo.py', kwargs['settings_file'], '--model_selection', model_selection] + cmd_args
        proc = subprocess.Popen(command, stdout=log_fp, stderr=subprocess.STDOUT)
    #
    return proc

def check_running_status(proc_dict):
    completed = False
    num_process = len(proc_dict)
    num_completed = 0
    tqdm_obj = tqdm.tqdm(total=num_process, desc='run status (R->Running, C->Completed): ')
    while not completed:
        status_dict = {proc_key: ("R" if proc else "C") for proc_key, proc in proc_dict.items()}
        for proc_key, proc in proc_dict.items():
            if proc is not None:
                exit_code = proc.returncode
                try:
                    out_ret, err_ret = proc.communicate(timeout=1.0)
                except subprocess.TimeoutExpired as ex:
                    pass
                else:
                    proc_dict[proc_key] = None
                    tqdm_obj.update()
                    tqdm_obj.set_postfix(status_dict)
                #
            #
        #
        num_completed = sum([proc is None for proc in proc_dict.values()])
        if num_completed >= num_process:
            completed = True
        #
    #
    return completed


if __name__ == '__main__':
    print(f'argv: {sys.argv}')
    # the cwd must be the root of the respository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #

    parser = get_arg_parser()
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

    with open(settings.models_list_file, 'rt') as list_fp:
        proc_dict = dict()
        for model_entry in list_fp:
            model_entry = model_entry.rstrip()
            model_entry = model_entry.split(' ')
            model_selection = model_entry[0]
            run_dir = model_entry[1]
            proc = run_one_model(kwargs, model_selection, run_dir)
            proc_dict.update({model_selection: proc})
        #
    #
    check_running_status(proc_dict)

    print("benchmark modelzoo parallel - COMPLETED")
