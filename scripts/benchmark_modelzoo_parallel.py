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
import copy
import os
import sys
import argparse
import warnings
import subprocess
import tqdm
import time
import functools

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
    parser.add_argument('--dataset_loading', type=utils.str_to_bool, default=True)
    parser.add_argument('--parallel_devices', type=utils.int_or_none)
    parser.add_argument('--parallel_processes', type=int)
    parser.add_argument('--fast_calibration_factor', type=utils.float_or_none)
    parser.add_argument('--experimental_models', type=utils.str_to_bool)
    parser.add_argument('--additional_models', type=utils.str_to_bool)
    parser.add_argument('--enable_logging', type=utils.str_to_bool)
    parser.add_argument('--models_list_file', type=str, default=None)
    parser.add_argument('--separate_import_inference', type=utils.str_to_bool, default=True)
    parser.add_argument('--overall_timeout', type=utils.float_or_none)
    parser.add_argument('--instance_timeout', type=utils.float_or_none)
    return parser


def run_one_model(entry_idx, kwargs, parallel_processes, model_selection, run_dir, enable_logging, run_import, run_inference):
    if kwargs['parallel_devices'] in (None, 0):
        parallel_devices = [0]
    else:
        parallel_devices = range(kwargs['parallel_devices']) if isinstance(kwargs['parallel_devices'], int) \
            else kwargs['parallel_devices']
    #

    cmd_args = []
    for key, value in kwargs.items():
        if key not in ('settings_file', 'models_list_file', 'parallel_processes', 'parallel_devices'):
            cmd_args += [f'--{key}']
            cmd_args += [f'{value}']
        #
    #

    # only relevant for compilation and when using tidl-tools with GPU/CUDA support
    num_devices = len(parallel_devices)
    parallel_device = parallel_devices[entry_idx%num_devices]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(parallel_device)

    # benchmark script
    command = ['python3',  './scripts/benchmark_modelzoo.py', kwargs['settings_file']]
    # add additional commanline arguments passed to this script
    command += cmd_args
    # specify which model(s) to run
    command += ['--model_selection', model_selection]
    # additional process helps with stability - even if a model compilation crashes, it won't affect the main program.
    # but, since we open a process with subprocess.Popen here, there is no need for the underlying python script to open a process inside
    command += ['--parallel_processes', '0']
    # which device should this be run on
    # relevant only if this is using the gpu/cuda tidl-tools and if there are multiple GPUs
    command += ['--parallel_devices', "1"]
    # logging is done by capturing the stdout and stderr to a file here - no need to enable logging to file inside
    command += ['--enable_logging', '0']
    # import and/or inference
    command += ['--run_import', f'{run_import}']
    command += ['--run_inference', f'{run_inference}']

    os.makedirs(run_dir, exist_ok=True)

    if parallel_processes and enable_logging:
        log_filename = os.path.join(run_dir, 'run.log')
        with open(log_filename, 'a') as log_fp:
            proc = subprocess.Popen(command, stdout=log_fp, stderr=log_fp)
        #
    elif parallel_processes:
        proc = subprocess.Popen(command)
    else:
        os.system(' '.join(command))
        proc = None
    #
    return proc


if __name__ == '__main__':
    print(f'argv: {sys.argv}')
    # the cwd must be the root of the respository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #

    parser = get_arg_parser()
    args = parser.parse_args()

    kwargs = vars(args)
    if 'session_type_dict' in kwargs:
        kwargs['session_type_dict'] = utils.str_to_dict(kwargs['session_type_dict'])
    #

    ####################################################################
    # create list of models to run
    kwargs_copy = copy.deepcopy(kwargs)
    kwargs_copy['dataset_loading'] = False
    settings = config_settings.ConfigSettings(args.settings_file, **kwargs_copy)
    print(f'settings: {settings}')
    sys.stdout.flush()

    ####################################################################
    try:
        if settings.target_machine == 'pc' and settings.parallel_devices is None:
            print(f"INFO: model compilation in PC can use cuda gpus (setup using setup_pc_gpu.sh)")
            nvidia_smi_command = 'nvidia-smi --list-gpus | wc -l'
            proc = subprocess.Popen([nvidia_smi_command], stdout=subprocess.PIPE, shell=True)
            out_ret, err_ret = proc.communicate()
            num_cuda_gpus = int(out_ret)
            print(f'INFO: - setting parallel_devices to the number of cuda gpus found: {num_cuda_gpus}')
            settings.parallel_devices = kwargs['parallel_devices'] = num_cuda_gpus
        #
    except:
        print("INFO: - could not find cuda gpus - parallel_devices will not be used.")
        settings.parallel_devices = kwargs['parallel_devices'] = None
    #

    ####################################################################
    work_dir = os.path.join(settings.modelartifacts_path, f'{settings.tensor_bits}bits')
    print(f'INFO: work_dir - {work_dir}')

    if kwargs['models_list_file'] is None:
        # make sure the folder exists
        os.makedirs(settings.modelartifacts_path, exist_ok=True)
        # create list file
        kwargs['models_list_file'] = os.path.join(settings.modelartifacts_path, "models_list.txt")
        # get a dict of model configs
        pipeline_configs = interfaces.get_configs(settings, work_dir)
        # filter the configs
        pipeline_configs = pipelines.PipelineRunner(settings, pipeline_configs).get_pipeline_configs()
        model_keys = pipeline_configs.keys()
        # now write to the list file
        with open(kwargs['models_list_file'], "w") as fp:
            for model_key in model_keys:
                run_dir = pipeline_configs[model_key]['session'].kwargs['run_dir']
                fp.write(f"{model_key} {run_dir}\n")
            #
        #
    #

    ####################################################################
    # get the list of models and run
    with open(kwargs['models_list_file'], 'rt') as list_fp:
        model_entries = [model_entry.rstrip() for model_entry in list_fp]
    #

    parallel_processes = kwargs.pop('parallel_processes', 0) or settings.parallel_processes
    parallel_devices = kwargs['parallel_devices']
    overall_timeout = kwargs.pop('overall_timeout', None)
    instance_timeout = kwargs.pop('instance_timeout', None)
    separate_import_inference = kwargs.pop('separate_import_inference')
    parallel_subprocess = utils.ParallelSubProcess(parallel_processes=parallel_processes, parallel_devices=parallel_devices,
        overall_timeout=overall_timeout, instance_timeout=instance_timeout)
    sequential_process_list = []

    for entry_idx, model_entry in enumerate(model_entries):
        model_entry = model_entry.split(' ')
        model_selection = model_entry[0]
        run_dir = model_entry[1]
        task_list_for_model = []
        if separate_import_inference:
            if settings.run_import:
                proc_name = f'{model_selection}:import'
                run_import_task = functools.partial(run_one_model,
                    entry_idx, kwargs, parallel_processes, model_selection, run_dir, settings.enable_logging, True, False)
                task_list_for_model.append({'proc_name':proc_name, 'proc_func':run_import_task})
            #
            if settings.run_inference:
                proc_name = f'{model_selection}:infer'
                run_inference_task = functools.partial(run_one_model,
                    entry_idx, kwargs, parallel_processes, model_selection, run_dir, settings.enable_logging, False, True)
                task_list_for_model.append({'proc_name':proc_name, 'proc_func':run_inference_task})
            #
        else:
            proc_name = f'{model_selection}'
            run_task = functools.partial(run_one_model,
                entry_idx, kwargs, parallel_processes, model_selection, run_dir, settings.enable_logging, settings.run_import, settings.run_inference)
            task_list_for_model.append({'proc_name':proc_name, 'proc_func':run_task})
        #
        if parallel_processes:
            parallel_subprocess.enqueue(task_name=model_selection, task_list=task_list_for_model)
        else:
            sequential_process_list.append(task_list_for_model)
        #
    #

    if parallel_processes:
        parallel_subprocess.run()
    else:
        for proc_func_list in tqdm.tqdm(sequential_process_list):
            for proc_entry in proc_func_list:
                proc_func = proc_entry['proc_func']
                proc_func()
            #
        #
    #
