import os
import yaml

work_dir = './work_dirs/benchmark_accuracy/8bits'

results_yaml = os.path.join(work_dir, 'results.yaml')
with open(results_yaml) as rfp:
    results = yaml.safe_load(rfp)

results_csv = os.path.join(work_dir, 'results.csv')
with open(results_csv, 'w') as wfp:
    for pipeline_id, pipeline_params in results.items():
        if pipeline_params is not None:
            wfp.write(f'{pipeline_id}, {pipeline_params["session"]["model_path"]}\n')
        #
    #
#
