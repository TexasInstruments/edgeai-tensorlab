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

metric_keys = ['accuracy_top1%', 'accuracy_mean_iou%', 'accuracy_ap[.5:.95]%']
result_keys = ['num_subgraphs', 'infer_time_core_ms', 'ddr_transfer_mb', 'perfsim_time_ms', 'perfsim_ddr_transfer_mb', 'perfsim_gmacs']


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


def run_report(benchmark_dir, rewrite_results=True):
    work_dirs = glob.glob(f'{benchmark_dir}/*')
    work_dirs = [f for f in work_dirs if os.path.isdir(f)]
    work_dirs = [d for d in work_dirs if os.path.basename(d) in ['8bits', '16bits', '32bits']]

    results_collection = dict()
    for work_dir in work_dirs:
        results_yaml = os.path.join(work_dir, 'results.yaml')
        # generate results.yaml, aggregating results from all the artifacts across all work_dirs.
        if rewrite_results:
            run_rewrite_results(work_dir, results_yaml)
        #
        tensor_bits = os.path.split(work_dir)[-1]
        with open(results_yaml) as rfp:
            results = yaml.safe_load(rfp)
            results_collection[tensor_bits] = results
        #
    #
    if len(results_collection) == 0:
        print('no results found - no report to generate.')
        return
    #

    results_8bits = results_collection['8bits'] if '8bits' in results_collection else None
    results_16bits = results_collection['16bits'] if '16bits' in results_collection else None
    results_32bits = results_collection['32bits'] if '32bits' in results_collection else None
    results_anchor = results_8bits or results_16bits or results_32bits

    if results_anchor is None:
        print('no result found - cannot generate report.')
        return
    #

    results_collection = list()
    title_line = ['serial_num', 'model_id', 'runtime_name', 'task_type', 'input_resolution', 'model_path', 'metric_name',
                  'metric_8bits', 'metric_16bits', 'metric_float', 'metric_reference'] + \
                  result_keys + ['run_dir', 'artifact_name']

    results_collection.append(title_line)
    for serial_num, (artifact_id, pipeline_params_anchor) in enumerate(results_anchor.items()):
        model_id = pipeline_params_anchor['session']['model_id']
        results_line_dict = {title_key:None for title_key in title_line}
        results_line_dict['serial_num'] = serial_num+1
        results_line_dict['model_id'] = model_id

        if pipeline_params_anchor is not None:
            results_line_dict['runtime_name'] = pipeline_params_anchor['session']['session_name']
            preprocess_crop = pipeline_params_anchor['preprocess']['crop']
            results_line_dict['input_resolution'] = 'x'.join(map(str, preprocess_crop)) \
                if isinstance(preprocess_crop, (list,tuple)) else str(preprocess_crop)
            model_path = pipeline_params_anchor['session']['model_path']
            model_path = model_path[0] if isinstance(model_path, (list,tuple)) else model_path
            results_line_dict['model_path'] = os.path.basename(model_path)
            results_line_dict['task_type'] = pipeline_params_anchor['task_type'] \
                if 'task_type' in pipeline_params_anchor else None
        #

        metric_name, _, metric_reference = get_metric(pipeline_params_anchor)
        results_line_dict['metric_name'] = metric_name

        if results_8bits is not None:
            pipeline_params_8bits = results_8bits[artifact_id] if artifact_id in results_8bits else None
            _, metric_8bits, _ = get_metric(pipeline_params_8bits)
            results_line_dict['metric_8bits'] = metric_8bits
        #

        if results_16bits is not None:
            pipeline_params_16bits = results_16bits[artifact_id] if artifact_id in results_16bits else None
            _, metric_16bits, _ = get_metric(pipeline_params_16bits)
            results_line_dict['metric_16bits'] = metric_16bits
        #

        if results_32bits is not None:
            pipeline_params_32bits = results_32bits[artifact_id] if artifact_id in results_32bits else None
            _, metric_32bits, _ = get_metric(pipeline_params_32bits)
            results_line_dict['metric_float'] = metric_32bits
        #

        results_line_dict['metric_reference'] = metric_reference

        performance_line_dict = get_performance(pipeline_params_anchor)
        results_line_dict.update(performance_line_dict)

        run_dir = pipeline_params_anchor['session']['run_dir'] if pipeline_params_anchor is not None else None
        run_dir_basename = os.path.basename(run_dir)
        results_line_dict['run_dir'] = run_dir_basename if run_dir is not None else None

        artifact_id = '_'.join(run_dir_basename.split('_')[:2])
        artifact_name = utils.get_artifact_name(artifact_id)
        artifact_name = '_'.join(run_dir_basename.split('_')[1:]) if artifact_name is None else artifact_name
        results_line_dict['artifact_name'] = artifact_name

        results_collection.append(list(results_line_dict.values()))
    #

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    report_csv = os.path.join(benchmark_dir, f'report_{date}.csv')
    with open(report_csv, 'w') as wfp:
        for results_line in results_collection:
            results_line = [str(r) for r in results_line]
            results_str = ','.join(results_line)
            wfp.write(f'{results_str}\n')
        #
    #


def get_metric(pipeline_params):
    global metric_keys
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


def get_performance(pipeline_params):
    global result_keys
    performance_line_dict = {result_key:None for result_key in result_keys}
    if pipeline_params is not None:
        if 'result' in pipeline_params:
            result = pipeline_params['result']
            if result is not None:
                for result_key in result_keys:
                    result_word = result[result_key] if result_key in result else None
                    performance_line_dict[result_key] = result_word
                #
            #
        #
    #
    return performance_line_dict
