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
import datetime
import yaml
import glob

from .. import utils


def run_rewrite_results(work_dir, results_yaml):
    run_dirs = glob.glob(f'{work_dir}/*')
    run_dirs = [f for f in run_dirs if os.path.isdir(f)]
    results = {}
    for run_dir in run_dirs:
        try:
            result_yaml = os.path.join(run_dir, 'result.yaml')
            with open(result_yaml) as fp:
                result = yaml.safe_load(fp)
                model_id = result['session']['model_id']
                session_name = result['session']['session_name']
                artifact_id = f'{model_id}_{session_name}'
                results[artifact_id] = result
            #
        except:
            pass
        #
    #
    results = utils.sorted_dict(results)
    with open(results_yaml, 'w') as rfp:
        yaml.safe_dump(results, rfp)
    #
    return results


def run_report(settings, rewrite_results=True):
    benchmark_dir = settings.modelartifacts_path
    report_perfsim = settings.report_perfsim

    work_dirs = glob.glob(f'{benchmark_dir}/*')
    work_dirs = [f for f in work_dirs if os.path.isdir(f)]
    work_dirs = [d for d in work_dirs if 'bits' in os.path.basename(d)]
    work_dirs = sorted(work_dirs, reverse=True)
    work_dirs = [w for w in work_dirs if '32bits' not in w] + [w for w in work_dirs if '32bits' in w]

    settings_names = []
    results_max_len = 0
    results_max_id = 0
    results_max_name = None
    results_collection = dict()
    for work_id, work_dir in enumerate(work_dirs):
        results_yaml = os.path.join(work_dir, 'results.yaml')
        # generate results.yaml, aggregating results from all the artifacts across all work_dirs.
        if rewrite_results:
            run_rewrite_results(work_dir, results_yaml)
        #
        settings_name = os.path.split(work_dir)[-1]
        with open(results_yaml) as rfp:
            results = yaml.safe_load(rfp)
            results_collection[settings_name] = results
            if len(results) > results_max_len:
                results_max_len = len(results)
                results_max_id = work_id
                results_max_name = settings_name
            #
        #
        settings_names.append(settings_name)
    #
    if len(results_collection) == 0:
        print('no results found - no report to generate.')
        return
    #

    results_anchor = results_collection[results_max_name]
    if results_anchor is None:
        print('no result found - cannot generate report.')
        return
    #

    metric_keys = ['accuracy_top1%', 'accuracy_mean_iou%', 'accuracy_ap[.5:.95]%', 'accuracy_delta_1%', 'accuracy_ap_3d_moderate%']
    performance_keys = ['num_subgraphs', 'infer_time_core_ms', 'ddr_transfer_mb']
    if report_perfsim:
        performance_keys += ['perfsim_time_ms', 'perfsim_ddr_transfer_mb', 'perfsim_gmacs']
    #

    results_table = list()
    metric_title = ['metric_'+m for m in results_collection.keys()] + ['metric_reference']
    title_line = ['serial_num', 'model_id', 'runtime_name', 'task_type', 'input_resolution', 'model_path', 'metric_name'] + \
        metric_title + performance_keys + ['run_dir', 'artifact_name']

    results_table.append(title_line)
    for serial_num, (artifact_id, pipeline_params_anchor) in enumerate(results_anchor.items()):
        model_id = pipeline_params_anchor['session']['model_id']
        results_line_dict = {title_key:None for title_key in title_line}
        results_line_dict['serial_num'] = serial_num+1
        results_line_dict['model_id'] = model_id

        if pipeline_params_anchor is not None:
            results_line_dict['runtime_name'] = pipeline_params_anchor['session']['session_name']
            preprocess_crop = pipeline_params_anchor['preprocess'].get('crop',None)
            results_line_dict['input_resolution'] = 'x'.join(map(str, preprocess_crop)) \
                if isinstance(preprocess_crop, (list,tuple)) else str(preprocess_crop)
            model_path = pipeline_params_anchor['session']['model_path']
            model_path = model_path[0] if isinstance(model_path, (list,tuple)) else model_path
            results_line_dict['model_path'] = os.path.basename(model_path)
            results_line_dict['task_type'] = pipeline_params_anchor['task_type'] \
                if 'task_type' in pipeline_params_anchor else None
        #
        metric_name, _, metric_reference = get_metric(pipeline_params_anchor, metric_keys)
        results_line_dict['metric_name'] = metric_name

        # now populate results for each setting/work_id
        for work_id, (settings_name, param_results) in enumerate(results_collection.items()):
            param_result = param_results[artifact_id] if artifact_id in param_results else None
            _, metric_value, _ = get_metric(param_result, metric_keys)
            results_line_dict[metric_title[work_id]] = metric_value
        #
        results_line_dict['metric_reference'] = metric_reference

        performance_line_dict = get_performance(pipeline_params_anchor, performance_keys)
        results_line_dict.update(performance_line_dict)

        run_dir = pipeline_params_anchor['session']['run_dir'] if pipeline_params_anchor is not None else None
        run_dir_basename = os.path.basename(run_dir)
        results_line_dict['run_dir'] = run_dir_basename if run_dir is not None else None

        artifact_id = '_'.join(run_dir_basename.split('_')[:2])
        artifact_name = utils.get_artifact_name(artifact_id)
        artifact_name = '_'.join(run_dir_basename.split('_')[1:]) if artifact_name is None else artifact_name
        results_line_dict['artifact_name'] = artifact_name

        results_table.append(list(results_line_dict.values()))
    #

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    report_csv = os.path.join(benchmark_dir, f'report_{date}.csv')
    with open(report_csv, 'w') as wfp:
        for results_line in results_table:
            results_line = [str(r) for r in results_line]
            results_str = ','.join(results_line)
            wfp.write(f'{results_str}\n')
        #
    #


def get_metric(pipeline_params, metric_keys):
    metric_name = None
    metric = None
    metric_reference = None
    if pipeline_params is not None:
        if 'result' in pipeline_params and pipeline_params['result'] is not None:
            result = pipeline_params['result']
            for metric_key in metric_keys:
                if metric_key in result:
                    metric = result[metric_key]
                    metric_name = metric_key
                #
            #
        #
        if 'model_info' in pipeline_params and pipeline_params['model_info'] is not None:
            model_info = pipeline_params['model_info']
            metric_reference_dict = model_info['metric_reference']
            for metric_key in metric_keys:
                if metric_key in metric_reference_dict:
                    metric_reference = metric_reference_dict[metric_key]
                #
            #
        #
    #
    return metric_name, metric, metric_reference


def get_performance(pipeline_params, performance_keys):
    performance_line_dict = {result_key:None for result_key in performance_keys}
    if pipeline_params is not None:
        if 'result' in pipeline_params:
            result = pipeline_params['result']
            if result is not None:
                for result_key in performance_keys:
                    result_word = result[result_key] if result_key in result else None
                    performance_line_dict[result_key] = result_word
                #
            #
        #
    #
    return performance_line_dict
