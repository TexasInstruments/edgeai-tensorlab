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
import functools
import subprocess

from .. import utils, pipelines, config_settings, datasets, constants
from .get_configs import *


__all__ = ['run_benchmark_script']


def run_benchmark_script_one_model(settings_file, entry_idx, cmd_kwargs, parallel_processes, parallel_devices, model_selection, run_dir,
    log_filename, run_import, run_inference, benchmark_script=None):

    if benchmark_script is None:
        benchmark_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts/benchmark_modelzoo.py'))
    #

    if isinstance(parallel_devices, (list,tuple)):
        parallel_devices = parallel_devices
    elif parallel_devices in (None, 0):
        parallel_devices = [0]
    elif isinstance(parallel_devices, int):
        parallel_devices = list(range(parallel_devices))
    else:
        assert False, f'unknown value of settings.parallel_devices: {parallel_devices} in {__file__}'

    # only relevant for compilation and when using tidl-tools with GPU/CUDA support
    num_devices = len(parallel_devices)
    parallel_device = parallel_devices[entry_idx%num_devices]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(parallel_device)

    # specify which model(s) to run
    cmd_kwargs['model_selection'] = model_selection
    # additional process helps with stability - even if a model compilation crashes, it won't affect the main program.
    # but, since we open a process with subprocess.Popen here, there is no need for the underlying python script to open a process inside
    cmd_kwargs['parallel_processes'] = '0'
    # which device should this be run on
    # relevant only if this is using the gpu/cuda tidl-tools and if there are multiple GPUs
    cmd_kwargs['parallel_devices'] = '1'
    # import and/or inference
    cmd_kwargs['run_import'] = f'{run_import}'
    cmd_kwargs['run_inference'] = f'{run_inference}'

    # benchmark script
    command = ['python3',  benchmark_script, settings_file]
    for key, value in cmd_kwargs.items():
        command += ['--'+str(key), str(value)]
    #

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    if parallel_processes:
        with open(log_filename, 'a') as log_fp:
            proc = subprocess.Popen(command, stdout=log_fp, stderr=log_fp)
        #
    else:
        os.system(' '.join(command))
        proc = None
    #
    return proc


def run_benchmark_script(settings, model_entries, settings_file, cmd_kwargs,
        overall_timeout=None, instance_timeout=None, proc_error_regex_list=None, separate_import_inference=True):

    if proc_error_regex_list is None:
        proc_error_regex_list=constants.TIDL_FATAL_ERROR_LOGS_REGEX_LIST
    #

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
                run_import_task = functools.partial(run_benchmark_script_one_model, settings_file,
                    entry_idx, cmd_kwargs, settings.parallel_processes, settings.parallel_devices, model_selection, run_dir, log_filename, True, False)
                task_list_for_model.append({'proc_name':proc_name, 'proc_func':run_import_task, 'proc_log':log_filename, 'proc_error':proc_error_regex_list})
            #
            if settings.run_inference:
                proc_name = f'{model_selection}:infer'
                run_inference_task = functools.partial(run_benchmark_script_one_model, settings_file,
                    entry_idx, cmd_kwargs, settings.parallel_processes, settings.parallel_devices, model_selection, run_dir, log_filename, False, True)
                task_list_for_model.append({'proc_name':proc_name, 'proc_func':run_inference_task, 'proc_log':log_filename, 'proc_error':proc_error_regex_list})
            #
        else:
            proc_name = f'{model_selection}'
            run_task = functools.partial(run_benchmark_script_one_model, settings_file,
                entry_idx, cmd_kwargs, settings.parallel_processes, settings.parallel_devices, model_selection, run_dir, log_filename, settings.run_import, settings.run_inference)
            task_list_for_model.append({'proc_name':proc_name, 'proc_func':run_task, 'proc_log':log_filename, 'proc_error':proc_error_regex_list})
        #

        task_entries.update({model_selection:task_list_for_model})
    #

    process_runner = utils.ProcessRunner(
        parallel_processes=settings.parallel_processes,
        overall_timeout=overall_timeout, instance_timeout=instance_timeout)
    process_runner.run(task_entries)
