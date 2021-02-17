import functools

from .accuracy_pipeline import *
from .. import utils


def run(config, pipeline_configs, parallel_devices=None):
    for model_id, pipeline_config in pipeline_configs.items():
        pipeline_config['session'].set_param('model_id', model_id)
    #
    if parallel_devices is not None:
        _run_pipelines_parallel(config, pipeline_configs, parallel_devices)
    else:
        _run_pipelines_sequential(config, pipeline_configs)
    #


def _run_pipelines_sequential(config, pipeline_configs):
    # get the cwd so that we can continue even if exception occurs
    cwd = os.getcwd()
    results_list = []
    desc = ' '*120 + 'TASKS'
    for pipeline_config in utils.progress_step2(pipeline_configs.values(), desc=desc):
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
    desc = ' '*120 + 'TASKS'
    parallel_exec = utils.ParallelRun(num_processes=num_devices, desc=desc)
    for pipeline_index, pipeline_config in enumerate(pipeline_configs.values()):
        if _check_model_selection(config, pipeline_config):
            os.chdir(cwd)
            parallel_device = parallel_devices[pipeline_index % num_devices] if parallel_devices is not None else 0
            run_pipeline_bound_func = functools.partial(_run_pipeline, pipeline_config, parallel_device=parallel_device)
            parallel_exec.enqueue(run_pipeline_bound_func)
        #
    #
    results_list = parallel_exec.run()
    return results_list


def _check_model_selection(config, pipeline_config):
    if config.model_selection is not None:
        selected_model = False
        model_path = pipeline_config['session'].kwargs['model_path']
        model_id = pipeline_config['session'].kwargs['model_id']
        model_selection = utils.as_list(config.model_selection)
        for keyword in model_selection:
            if keyword in model_path:
                selected_model = True
            #
            if keyword in model_id:
                selected_model = True
            #
        #
    else:
        selected_model = True
    #
    return selected_model


def _run_pipeline(pipeline_config, parallel_device=None):
    if parallel_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(parallel_device)
    #
    result = {}
    pipeline_types = utils.as_list(pipeline_config['type'])
    try:
        for pipeline_type in pipeline_types:
            if pipeline_type == 'accuracy':
                # use with statement, so that the logger and other file resources are cleaned up
                with AccuracyPipeline(pipeline_config) as accuracy_pipeline:
                    accuracy_result = accuracy_pipeline.run()
                    result.update(accuracy_result)
                #
            elif pipeline_type == 'something':
                # this is just an example of how other pipelines can be implemented.
                # 'something' used here is not real and it is not supported
                with SomethingPipeline(pipeline_config) as something_pipeline:
                    something_result = something_pipeline.run()
                    result.update(something_result)
                #
            else:
                assert False, f'unknown pipeline: {pipeline_type}'
            #
    except Exception as e:
        print(f'\n{str(e)}')
    #
    return result

