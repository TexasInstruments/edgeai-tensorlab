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

from .. import utils, pipelines, config_settings, datasets
from .get_configs import *

__all__ = ['run_benchmark_config']


def run_benchmark_config_one_model(settings, entry_idx, proc_name, proc_func):
    if settings.parallel_processes and settings.parallel_devices:
        # only relevant for compilation and when using tidl-tools with GPU/CUDA support
        num_devices = len(settings.parallel_devices)
        parallel_device = settings.parallel_devices[entry_idx%num_devices]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(parallel_device)
    #
    proc = utils.ProcessWtihQueue(name=proc_name, target=proc_func)
    proc.start()
    return proc


def run_benchmark_config(settings, work_dir, pipeline_configs=None, modify_pipelines_func=None,
    overall_timeout=None, instance_timeout=None, proc_error_regex_list=None, separate_import_inference=True):
    
    # verify that targt device is correct
    if settings.target_device is not None and 'TIDL_TOOLS_PATH' in os.environ and \
            os.environ['TIDL_TOOLS_PATH'] is not None:
        assert settings.target_device in os.environ['TIDL_TOOLS_PATH'], \
            f'found target_device in settings: {settings.target_device} ' \
            f'but it does not seem to match tidl_tools path: {os.environ["TIDL_TOOLS_PATH"]}'
    #

    # get the default configs if pipeline_configs is not given from outside
    if pipeline_configs is None:
        # get a dict of model configs
        pipeline_configs = get_configs(settings, work_dir)
    #


    # create the pipeline_runner which will manage the sessions.
    pipeline_runner = pipelines.PipelineRunner(settings, pipeline_configs=pipeline_configs)

    ############################################################################
    # at this point, pipeline_runner.pipeline_configs is a dictionary that has the selected configs

    # Note: to manually slice and select a subset of configs, slice it this way (just an example)
    # import itertools
    # pipeline_runner.pipeline_configs = dict(itertools.islice(pipeline_runner.pipeline_configs.items(), 10, 20))

    # some examples of accessing params from it - here 0th entry is used an example.
    # pipeline_config = pipeline_runner.pipeline_configs.values()[0]
    # pipeline_config['preprocess'].get_param('resize') gives the resize dimension
    # pipeline_config['preprocess'].get_param('crop') gives the crop dimension
    # pipeline_config['session'].get_param('run_dir') gives the folder where artifacts are located
    ############################################################################

    if modify_pipelines_func is not None:
        pipeline_runner.pipeline_configs = modify_pipelines_func(pipeline_runner.pipeline_configs)
    #

    # print some info
    run_dirs = [pipeline_config['session'].get_param('run_dir') for model_key, pipeline_config \
                in pipeline_runner.pipeline_configs.items()]
    run_dirs = [os.path.basename(run_dir) for run_dir in run_dirs]
    print(f'configs to run: {run_dirs}')
    print(f'number of configs: {len(pipeline_runner.pipeline_configs)}')
    sys.stdout.flush()

    task_entries = pipeline_runner.get_tasks(separate_import_inference=separate_import_inference)

    if settings.parallel_processes:
        for task_entry_idx, (task_name, task_list) in enumerate(task_entries.items()):
            for proc_entry in task_list:
                proc_name = proc_entry['proc_name']
                proc_func = proc_entry['proc_func']
                proc_func = functools.partial(run_benchmark_config_one_model, settings, task_entry_idx, proc_name, proc_func)
                proc_entry['proc_func'] =  proc_func
            #
        #
    #

    # now actually run the configs
    if settings.parallel_processes:
        process_runner = utils.ProcessRunner(
            parallel_processes=settings.parallel_processes, parallel_devices=settings.parallel_devices,
            overall_timeout=overall_timeout, instance_timeout=instance_timeout)
        return process_runner.run(task_entries)        
    else:
        return pipeline_runner.run(task_entries)
