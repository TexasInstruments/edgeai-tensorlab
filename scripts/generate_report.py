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
import glob


metric_keys = ['accuracy_top1%', 'accuracy_mean_iou%', 'accuracy_ap[.5:.95]%']
result_keys = ['num_subgraphs', 'infer_time_core_ms', 'perfsim_ddr_transfer_mb', 'perfsim_gmacs']


def main():
    benchmark_dir = './work_dirs/benchmark_accuracy'
    work_dirs = glob.glob(f'{benchmark_dir}/*')
    work_dirs = [f for f in work_dirs if os.path.isdir(f)]

    results_collection = dict()
    for work_dir in work_dirs:
        tidl_tensor_bits = os.path.split(work_dir)[-1]
        results_yaml = os.path.join(work_dir, 'results.yaml')
        with open(results_yaml) as rfp:
            results = yaml.safe_load(rfp)
            results_collection[tidl_tensor_bits] = results
        #
    #

    results_8bits = results_collection['8bits']
    results_16bits = results_collection['16bits']
    results_32bits = results_collection['32bits']
    results_collection = list()
    title_line = ['model_id', 'run_time', 'task_type', 'input_resolution', 'model_path', 'metric_name',
                  'metric_8bits', 'metric_16bits', 'metric_float', 'metric_reference'] + \
                 result_keys + ['run_dir']
    results_collection.append(title_line)
    for pipeline_id, pipeline_params_8bits in results_8bits.items():
        results_line_dict = {title_key:None for title_key in title_line}
        results_line_dict['model_id'] = pipeline_id

        metric_name, metric_8bits, metric_reference = get_metric(pipeline_params_8bits)
        if pipeline_params_8bits is not None:
            results_line_dict['run_time'] = pipeline_params_8bits['session']['session_name']
            preprocess_crop = pipeline_params_8bits['preprocess']['crop']
            results_line_dict['input_resolution'] = 'x'.join(map(str, preprocess_crop)) \
                if isinstance(preprocess_crop, (list,tuple)) else str(preprocess_crop)
            model_path = pipeline_params_8bits['session']['model_path']
            model_path = model_path[0] if isinstance(model_path, (list,tuple)) else model_path
            results_line_dict['model_path'] = os.path.basename(model_path)
            results_line_dict['task_type'] = pipeline_params_8bits['task_type'] \
                if 'task_type' in pipeline_params_8bits else None
        #
        results_line_dict['metric_name'] = metric_name
        results_line_dict['metric_8bits'] = metric_8bits

        pipeline_params_16bits = results_16bits[pipeline_id] if pipeline_id in results_16bits else None
        _, metric_16bits, _ = get_metric(pipeline_params_16bits)
        results_line_dict['metric_16bits'] = metric_16bits

        pipeline_params_32bits = results_32bits[pipeline_id] if pipeline_id in results_32bits else None
        _, metric_32bits, _ = get_metric(pipeline_params_32bits)
        results_line_dict['metric_float'] = metric_32bits

        results_line_dict['metric_reference'] = metric_reference

        performance_line_dict = get_performance(pipeline_params_8bits)
        results_line_dict.update(performance_line_dict)

        run_dir = pipeline_params_8bits['session']['run_dir'] if pipeline_params_8bits is not None else None
        results_line_dict['run_dir'] = os.path.basename(run_dir) if run_dir is not None else None

        results_collection.append(list(results_line_dict.values()))
    #

    report_csv = os.path.join(benchmark_dir, 'report.csv')
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


if __name__ == '__main__':
    # the cwd must be the root of the repository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #
    main()

