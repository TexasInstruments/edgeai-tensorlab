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


def get_run_dirs(work_dir):
    run_dirs = glob.glob(f'{work_dir}/*')
    run_dirs = [f for f in run_dirs if os.path.isdir(f)]
    return run_dirs


def run_rewrite_results(work_dir, results_yaml):
    run_dirs = get_run_dirs(work_dir)
    results = {}
    for run_dir in run_dirs:
        result_yaml = os.path.join(run_dir, 'result.yaml')
        config_yaml = os.path.join(run_dir, 'config.yaml')
        result_or_config_yaml = None
        if os.path.exists(result_yaml):
            result_or_config_yaml = result_yaml
        elif os.path.exists(config_yaml):
            result_or_config_yaml = config_yaml
        #
        if result_or_config_yaml:
            with open(result_or_config_yaml) as fp:
                result = yaml.safe_load(fp)
                model_id = result['session']['model_id']
                session_name = result['session']['session_name']
                artifact_id = f'{model_id}_{session_name}'
                results[artifact_id] = result
            #
        #
    #
    results = utils.sorted_dict(results)
    with open(results_yaml, 'w') as rfp:
        yaml.safe_dump(results, rfp)
    #
    return results


def run_report(settings, rewrite_results=True, skip_pattern=None):
    report_perfsim = settings.report_perfsim

    if settings.target_device in (None, 'None'):
        benchmark_dir = os.path.dirname(settings.modelartifacts_path)
        target_device_dirs = os.listdir(benchmark_dir)
    else:
        benchmark_dir = os.path.dirname(settings.modelartifacts_path)
        target_device_dirs = [settings.target_device]
    #

    work_dirs = []
    for target_device in target_device_dirs:
        target_device_dir = os.path.join(benchmark_dir, target_device)
        target_work_dirs = glob.glob(f'{target_device_dir}/*')
        work_dirs += target_work_dirs
    #

    work_dirs = [f for f in work_dirs if os.path.isdir(f)]
    work_dirs = [d for d in work_dirs if 'bits' in os.path.basename(d)]
    work_dirs = sorted(work_dirs, reverse=True)
    work_dirs = [w for w in work_dirs if '32bits' in w] + [w for w in work_dirs if '32bits' not in w]

    work_dir_keys = []
    work_dir_results_max_len = 0
    work_dir_results_max_id = 0
    work_dir_results_max_name = None
    work_dir_results_max_path = None
    results_collection = dict()
    for work_id, work_dir in enumerate(work_dirs):
        results_yaml = os.path.join(work_dir, 'results.yaml')
        # generate results.yaml, aggregating results from all the artifacts across all work_dirs.
        if rewrite_results:
            run_rewrite_results(work_dir, results_yaml)
        #
        work_dir_splits = os.path.normpath(work_dir).split(os.sep)
        run_dirs = get_run_dirs(work_dir)
        work_dir_key = '_'.join(work_dir_splits[-2:])
        if skip_pattern is None or skip_pattern not in work_dir_key:
            with open(results_yaml) as rfp:
                results = yaml.safe_load(rfp)
                results_collection[work_dir_key] = results
                if len(run_dirs) > work_dir_results_max_len:
                    work_dir_results_max_len = len(run_dirs)
                    work_dir_results_max_id = work_id
                    work_dir_results_max_name = work_dir_key
                    work_dir_results_max_path = work_dir
                #
            #
            work_dir_keys.append(work_dir_key)
        #
    #
    if len(results_collection) == 0 or work_dir_results_max_name is None:
        print('no results found - no report to generate.')
        return
    #

    results_anchor = results_collection[work_dir_results_max_name]
    if results_anchor is None:
        print('no result found - cannot generate report.')
        return
    #

    print(f'results found for {work_dir_results_max_len} models')

    metric_keys = ['accuracy_top1%', 'accuracy_mean_iou%', 'accuracy_ap[.5:.95]%', 'accuracy_delta_1%',
                   'accuracy_ap_3d_moderate%', 'accuracy_add(s)_p1%', 'accuracy_localization%']
    performance_keys = ['num_subgraphs', 'infer_time_core_ms', 'ddr_transfer_mb']
    if report_perfsim:
        performance_keys += ['perfsim_time_ms', 'perfsim_ddr_transfer_mb', 'perfsim_gmacs']
    #

    results_table = dict()
    metric_title = [m+'_metric' for m in results_collection.keys()]
    performance_title = [m+'_'+p for m in results_collection.keys() for p in performance_keys]
    title_line = ['serial_num', 'model_id', 'runtime_name', 'task_type', 'dataset_name', 'run_dir', 'input_resolution', 'metric_name'] + \
        metric_title + performance_title + ['metric_reference'] + ['model_shortlist', 'model_path', 'artifact_name']

    run_dirs = get_run_dirs(work_dir_results_max_path)
    for serial_num, run_dir in enumerate(run_dirs):
        results_line_dict = {title_key:None for title_key in title_line}

        param_yaml = os.path.join(run_dir, 'param.yaml')
        compilation_done = os.path.exists(param_yaml)

        run_dir_basename = os.path.basename(run_dir)
        run_dir_splits = run_dir_basename.split('_')
        artifact_id = '_'.join(run_dir_splits[:2]) if len(run_dir_splits) > 1 else run_dir_splits[0]
        model_id = run_dir_splits[0]

        results_line_dict['model_id'] = model_id
        results_line_dict['serial_num'] = serial_num+1

        pipeline_params_anchor = results_anchor.get(artifact_id, None)
        if pipeline_params_anchor is not None:
            model_id_from_result = pipeline_params_anchor['session']['model_id']
            assert model_id == model_id_from_result, f"model_id={model_id} doesnt match model_id_from_result={model_id_from_result}"
            results_line_dict['runtime_name'] = pipeline_params_anchor['session']['session_name']
            preprocess_crop = pipeline_params_anchor['preprocess'].get('crop',None)
            results_line_dict['input_resolution'] = 'x'.join(map(str, preprocess_crop)) \
                if isinstance(preprocess_crop, (list,tuple)) else str(preprocess_crop)
            model_path = pipeline_params_anchor['session']['model_path']
            model_path = model_path[0] if isinstance(model_path, (list,tuple)) else model_path
            results_line_dict['model_path'] = os.path.basename(model_path)
            results_line_dict['task_type'] = pipeline_params_anchor['task_type'] \
                if 'task_type' in pipeline_params_anchor else None
            results_line_dict['model_shortlist'] = pipeline_params_anchor['model_info'].get('model_shortlist', None)
            results_line_dict['dataset_name'] = pipeline_params_anchor.get('dataset_category', None)
        #
        metric_name, _, metric_reference = get_metric(pipeline_params_anchor, metric_keys, compilation_done)
        results_line_dict['metric_name'] = metric_name
        results_line_dict['metric_reference'] = metric_reference

        # now populate results for each setting/work_id
        for work_id, (work_dir_key, param_results) in enumerate(results_collection.items()):
            param_result = param_results[artifact_id] if artifact_id in param_results else None
            _, metric_value, _ = get_metric(param_result, metric_keys, compilation_done)
            results_line_dict[metric_title[work_id]] = metric_value
            performance_line_dict = get_performance(param_result, performance_keys)
            if performance_line_dict is not None:
                performance_line_dict = {work_dir_key+'_'+k:v for k, v in performance_line_dict.items()}
                results_line_dict.update(performance_line_dict)
            #
        #

        results_line_dict['run_dir'] = run_dir_basename

        artifact_id = '_'.join(run_dir_basename.split('_')[:2])
        artifact_name = utils.get_artifact_name(artifact_id)
        artifact_name = '_'.join(run_dir_basename.split('_')[1:]) if artifact_name is None else artifact_name
        results_line_dict['artifact_name'] = artifact_name

        results_table[artifact_id] = results_line_dict
    #

    # sort the results table based on keys
    results_table = {k:v for k, v in sorted(results_table.items(), key=lambda item: item[0])}

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    report_csv = os.path.join(benchmark_dir, f'report_{date}.csv')
    with open(report_csv, 'w') as wfp:
        title_str = ','.join(title_line)
        wfp.write(f'{title_str}\n')
        for results_key, results_line in results_table.items():
            results_keys = list(results_line.keys())
            # make sure reslut keys are same as title_line
            assert set(title_line) == set(results_keys), f'result keys do not match the title - expected: {title_line} obtained: {results_keys}'
            results_line = [str(r) for r in results_line.values()]
            results_str = ','.join(results_line)
            wfp.write(f'{results_str}\n')
        #
    #


def get_metric(pipeline_params, metric_keys, compilation_done):
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
        elif compilation_done is True:
            metric = 'no_inference'
        elif compilation_done is False:
            metric = 'no_compilation'
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
