import functools

from .accuracy_pipeline import *
from .. import utils


def run(config, pipeline_configs, parallel_devices=None):
    if parallel_devices is not None:
        _run_pipelines_parallel(config, pipeline_configs, parallel_devices)
    else:
        _run_pipelines_sequential(config, pipeline_configs)
    #


def _run_pipelines_sequential(config, pipeline_configs):
    # get the cwd so that we can continue even if exception occurs
    cwd = os.getcwd()
    results_list = []
    for pipeline_config in utils.progress_step(pipeline_configs, desc='tasks'):
        if _check_model_selection(config, pipeline_config):
            os.chdir(cwd)
            result = _run_pipeline(pipeline_config)
            results_list.append(result)
        #
    #
    return results_list


def _run_pipelines_parallel(config, pipeline_configs, parallel_devices=None):
    # get the cwd so that we can continue even if exception occurs
    cwd = os.getcwd()
    num_devices = len(parallel_devices) if parallel_devices is not None else 0
    parallel_exec = utils.ParallelRun(num_processes=num_devices)
    for model_idx, pipeline_config in enumerate(pipeline_configs):
        if _check_model_selection(config, pipeline_config):
            os.chdir(cwd)
            parallel_device = parallel_devices[model_idx % num_devices] if parallel_devices is not None else 0
            run_pipeline_bound_func = functools.partial(_run_pipeline, pipeline_config, parallel_device=parallel_device)
            parallel_exec.enqueue(run_pipeline_bound_func)
        #
    #
    results_list = parallel_exec.run()
    return results_list


def _check_model_selection(config, pipeline_config):
    selected_model = True
    if config.model_selection is not None:
        model_path = pipeline_config['session'].kwargs['model_path']
        model_selection = utils.as_list(config.model_selection)
        for keyword in model_selection:
            if keyword in model_path:
                selected_model = True
            #
        #
    #
    return selected_model


def _run_pipeline(pipeline_config, parallel_device=None):
    if parallel_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(parallel_device)
    #

    result = None
    try:
        pipeline_type = pipeline_config['type']
        if pipeline_type == 'accuracy':
            # use with statement, so that the logger and other file resources are cleaned up
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

