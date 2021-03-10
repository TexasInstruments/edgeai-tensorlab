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
import glob
import pickle
import yaml
from .. import utils


def collect_results(settings, work_dir, pipeline_configs, print_results=True):
    results = {}
    for pipeline_id, pipeline_config in pipeline_configs.items():
        # collect the result of the pipeline
        result_dict = collect_result(settings, pipeline_config)
        # print the result if necessary
        if print_results:
            print(f'{pipeline_id}: {utils.pretty_object(get_result(result_dict))}')
        #
        if isinstance(result_dict, dict) and 'result' in result_dict:
            # if the result is an an entry in this dict, then this is already the full dict
            param_result = result_dict
            # if you want to override the params fetched from run_dir with new params
            # param_result = {'result': result_dict['result']}
            # param_dict = collect_param(settings, pipeline_config)
            # param_result.update(param_dict)
        else:
            param_result = {'result': result_dict}
            param_dict = collect_param(settings, pipeline_config)
            param_result.update(param_dict)
        #
        results.update({pipeline_id: param_result})
    #

    # sort the results based on keys
    results_keys = sorted(results.keys())
    results = {k: results[k] for k in results_keys}

    # for logging
    results = utils.pretty_object(results)
    if settings.enable_logging:
        results_yaml_filename = os.path.join(work_dir, 'results.yaml')
        with open(results_yaml_filename, 'w') as fp:
            yaml.safe_dump(results, fp, sort_keys=False)
        #
        # TODO: deprecate this pkl file format later
        results_pkl_filename = os.path.join(work_dir, 'results.pkl')
        with open(results_pkl_filename, 'wb') as fp:
            # for backward compatibility, the pkl file has a list instead of dict
            pickle.dump(list(results.values()), fp)
        #
    #
    return results


def collect_param(settings, pipeline_config):
    pipeline_param = {}
    for pipeline_stage_name, pipeline_stage in pipeline_config.items():
        if hasattr(pipeline_stage, 'get_params'):
            kwargs = pipeline_stage.get_params()
        else:
            kwargs = pipeline_stage
        #
        if kwargs is not None:
            pipeline_param.update({pipeline_stage_name:kwargs})
        #
    #
    return pipeline_param


def collect_result(settings, pipeline_config):
    param_result = None
    run_dir = pipeline_config['session'].get_param('run_dir')
    if not (os.path.exists(run_dir) and os.path.isdir(run_dir)):
        return param_result
    #
    param_result = None
    if param_result is None:
        try:
            # TODO: deprecate this pkl file format later
            pkl_filename = f'{run_dir}/result.pkl'
            with open(pkl_filename, 'rb') as pkl_fp:
                param_result = pickle.load(pkl_fp)
            #
        except:
            pass
        #
    #
    if param_result is None:
        try:
            yaml_filename = f'{run_dir}/result.yaml'
            with open(yaml_filename, 'r') as yaml_fp:
                param_result = yaml.safe_load(yaml_fp)
            #
        except:
            # yaml_filename = f'{run_dir}/result.yaml'
            # with open(yaml_filename, 'w') as yaml_fp:
            #     yaml.safe_dump(utils.pretty_object(param_result), yaml_fp)
            # #
            pass
        #
    #
    if param_result is not None:
        if isinstance(param_result, dict) and 'result' in param_result:
            param_result['result'] = correct_result(param_result['result'])
        else:
            param_result = correct_result(param_result)
        #
    #
    return param_result


def correct_result(param_result):
    result = get_result(param_result)
    if result is not None and 'inference_path' in result:
        result['infer_path'] = result['inference_path']
        del result['inference_path']
    #
    if isinstance(param_result, dict) and 'result' in param_result:
        param_result['result'] = result
    #
    return result


def get_result(param_result):
    if isinstance(param_result, dict) and 'result' in param_result:
        # if this dict has param, then result will be an entry in it
        result = param_result['result']
    else:
        # else this whole dict is the result
        result = param_result
    #
    return result
