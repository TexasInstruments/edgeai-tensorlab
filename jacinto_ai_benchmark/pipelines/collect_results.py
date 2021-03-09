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
        result = collect_result(settings, pipeline_config)
        results.update({pipeline_id:result})
    #

    # sort the results
    results = {k:v for k,v in sorted(results.items())}

    # for logging and printing
    results = utils.round_dicts(results)
    if settings.enable_logging:
        results_yaml_filename = os.path.join(work_dir, 'results.yaml')
        with open(results_yaml_filename, 'w') as fp:
            yaml.safe_dump(results, fp)
        #
        # TODO: deprecate this pkl file format later
        results_pkl_filename = os.path.join(work_dir, 'results.pkl')
        with open(results_pkl_filename, 'wb') as fp:
            # for backward compatibility, the pkl file has a list instead of dict
            pickle.dump(list(results.values()), fp)
        #
    #
    if print_results:
        for result_id, result in results.items():
            print(f'{result_id}:{result}')
        #
    #
    return results


def collect_result(settings, pipeline_config):
    result = None
    run_dir = pipeline_config['session'].get_param('run_dir')
    if not (os.path.exists(run_dir) and os.path.isdir(run_dir)):
        return result
    #
    result = None
    try:
        yaml_filename = f'{run_dir}/result.yaml'
        with open(yaml_filename, 'rb') as yaml_fp:
            result = yaml.safe_load(yaml_fp)
        #
    except:
        pass
    #
    # TODO: deprecate this pkl file format later
    if result is None:
        try:
            pkl_filename = f'{run_dir}/result.pkl'
            with open(pkl_filename, 'rb') as pkl_fp:
                result = pickle.load(pkl_fp)
            #
        except:
            pass
        #
    #
    if result is not None:
        result = correct_result(result)
    #
    return result


def correct_result(result):
    if 'inference_path' in result:
        result['infer_path'] = result['inference_path']
        del result['inference_path']
    #
    return result