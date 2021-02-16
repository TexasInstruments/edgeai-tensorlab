import functools

from .accuracy_pipeline import *
from .. import utils


def run(pipeline_configs, parallel_devices=None):
    if parallel_devices is not None:
        _run_pipelines_parallel(pipeline_configs, parallel_devices)
    else:
        _run_pipelines_sequential(pipeline_configs)
    #


def _run_pipelines_sequential(pipeline_configs):
    # get the cwd so that we can continue even if exception occurs
    cwd = os.getcwd()
    results_list = []
    for pipeline_config in utils.progress_step(pipeline_configs, desc='tasks'):
        os.chdir(cwd)
        result = _run_pipeline(pipeline_config)
        results_list.append(result)
    #
    return results_list


def _run_pipelines_parallel(pipeline_configs, parallel_devices=None):
    # get the cwd so that we can continue even if exception occurs
    cwd = os.getcwd()
    num_devices = len(parallel_devices) if parallel_devices is not None else 0
    parallel_exec = utils.ParallelRun(num_processes=num_devices)
    for model_idx, pipeline_config in enumerate(pipeline_configs):
        os.chdir(cwd)
        parallel_device = parallel_devices[model_idx % num_devices] if parallel_devices is not None else 0
        run_pipeline_bound_func = functools.partial(_run_pipeline, pipeline_config, parallel_device=parallel_device)
        parallel_exec.enqueue(run_pipeline_bound_func)
    #
    results_list = parallel_exec.run()
    return results_list


def _run_pipeline(pipeline_config, parallel_device=None):
    if parallel_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(parallel_device)
    #

    result = None
    try:
        pipeline_type = pipeline_config['type']
        if pipeline_type == 'accuracy':
            # use with statement, so that the logger and other file resources are cleanedup
            with AccuracyPipeline(pipeline_config) as accuracy_pipeline:
                result = accuracy_pipeline.run()
            #
        else:
            assert False, f'unknown pipeline: {pipeline_type}'
        #
    except Exception as e:
        print(f'\n{str(e)}')
    #

    # perfsim_dict = None
    # try:
    #     perfsim_dict = pipeline_config['session'].perfsim_data() if perfsim else None
    # except Exception as e:
    #     print(f'\n{str(e)}')
    # #
    # if perfsim_dict is not None:
    #     results.update(perfsim_dict)
    # #

    return result

