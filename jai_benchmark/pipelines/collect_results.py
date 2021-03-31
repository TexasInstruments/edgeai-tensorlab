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
import yaml
from .. import utils


def collect_results(settings, work_dir, pipeline_configs, print_results=True, update_params=False):
    param_results = {}
    for pipeline_id, pipeline_config in pipeline_configs.items():
        session_name = pipeline_config['session']['session_name']
        artifact_id = f'{pipeline_id}_{session_name}'
        # collect the result of the pipeline
        param_result = collect_result(settings, pipeline_config)
        # print the result if necessary
        if print_results:
            print(f'{artifact_id}: {utils.pretty_object(get_result(param_result))}')
        #
        # if needed the params in result can be updated here
        if update_params and param_result is not None:
            param_dict = utils.pretty_object(pipeline_config)
            param_result.update(param_dict)
        #
        param_results.update({artifact_id: param_result})
    #
    # sort the results based on keys
    param_results_keys = sorted(param_results.keys())
    param_results = {k: param_results[k] for k in param_results_keys}
    # for logging
    param_results = utils.pretty_object(param_results)
    if settings.enable_logging:
        with open(os.path.join(work_dir, 'results.yaml'), 'w') as fp:
            yaml.safe_dump(param_results, fp, sort_keys=False)
        #
    #
    return param_results


def collect_result(settings, pipeline_config):
    param_result = None
    run_dir = pipeline_config['session'].get_param('run_dir')
    if not (os.path.exists(run_dir) and os.path.isdir(run_dir)):
        return param_result
    #
    try:
        yaml_filename = f'{run_dir}/result.yaml'
        with open(yaml_filename, 'r') as yaml_fp:
            param_result = yaml.safe_load(yaml_fp)
        #
    except:
        param_result = None
    #
    if param_result is not None:
        param_result = correct_result(param_result)
    #
    return param_result


def get_result(param_result):
    if isinstance(param_result, dict) and 'result' in param_result:
        # if this dict has param, then result will be an entry in it
        result = param_result['result']
    else:
        # else this whole dict is the result
        result = param_result
    #
    return result


def correct_result(param_result):
    if param_result is not None:
        result = get_result(param_result)
        if result is not None and 'inference_path' in result:
            result['infer_path'] = result['inference_path']
            del result['inference_path']
        #
        if isinstance(param_result, dict) and 'result' in param_result:
            param_result['result'] = result
        #
    #
    return param_result
