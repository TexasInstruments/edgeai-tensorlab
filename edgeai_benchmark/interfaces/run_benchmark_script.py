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
import subprocess

from .. import utils, pipelines, config_settings, datasets
from .get_configs import *


__all__ = ['run_benchmark_script']


def run_benchmark_script_one_model(entry_idx, kwargs, parallel_processes, model_selection, run_dir,
    enable_logging, log_filename, run_import, run_inference, benchmark_script=None):

    if benchmark_script is None:
        benchmark_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts/benchmark_modelzoo.py'))
    #

    if kwargs['parallel_devices'] in (None, 0):
        parallel_devices = [0]
    else:
        parallel_devices = range(kwargs['parallel_devices']) if isinstance(kwargs['parallel_devices'], int) \
            else kwargs['parallel_devices']
    #

    cmd_args = []
    for key, value in kwargs.items():
        if key not in ['settings_file']:
            cmd_args += [f'--{key}']
            cmd_args += [f'{value}']
        #
    #

    # only relevant for compilation and when using tidl-tools with GPU/CUDA support
    num_devices = len(parallel_devices)
    parallel_device = parallel_devices[entry_idx%num_devices]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(parallel_device)

    # benchmark script
    command = ['python3',  benchmark_script, kwargs['settings_file']]
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

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    if parallel_processes and enable_logging:
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


def run_benchmark_script(settings, model_entries, kwargs_dict, parallel_processes, parallel_devices,
        overall_timeout, instance_timeout, proc_error_regex_list, separate_import_inference=True):

    process_runner = pipelines.ProcessRunner(settings, pipeline_configs=None,
        parallel_processes=parallel_processes, parallel_devices=parallel_devices,
        overall_timeout=overall_timeout, instance_timeout=instance_timeout,
        with_subprocess=True)

    task_entries = {}
    for entry_idx, model_entry in enumerate(model_entries):
        model_entry = model_entry.split(' ')
        model_selection = model_entry[0]
        run_dir = model_entry[1]
        task_list_for_model = []
        log_filename = os.path.join(run_dir, 'run.log')

        if separate_import_inference:
            if settings.run_import:
                proc_name = f'{model_selection}:import'
                run_import_task = functools.partial(run_benchmark_script_one_model,
                    entry_idx, kwargs_dict, parallel_processes, model_selection, run_dir, settings.enable_logging, log_filename, True, False)
                task_list_for_model.append({'proc_name':proc_name, 'proc_func':run_import_task, 'proc_log':log_filename, 'proc_error':proc_error_regex_list})
            #
            if settings.run_inference:
                proc_name = f'{model_selection}:infer'
                run_inference_task = functools.partial(run_benchmark_script_one_model,
                    entry_idx, kwargs_dict, parallel_processes, model_selection, run_dir, settings.enable_logging, log_filename, False, True)
                task_list_for_model.append({'proc_name':proc_name, 'proc_func':run_inference_task, 'proc_log':log_filename, 'proc_error':proc_error_regex_list})
            #
        else:
            proc_name = f'{model_selection}'
            run_task = functools.partial(run_benchmark_script_one_model,
                entry_idx, kwargs_dict, parallel_processes, model_selection, run_dir, settings.enable_logging, log_filename, settings.run_import, settings.run_inference)
            task_list_for_model.append({'proc_name':proc_name, 'proc_func':run_task, 'proc_log':log_filename, 'proc_error':proc_error_regex_list})
        #

        task_entries.update({model_selection:task_list_for_model})
    #

    process_runner.run(task_entries)
