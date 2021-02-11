import os
import functools

from .accuracy_pipeline import *
from .. import utils


def run(pipeline_configs, perfsim=False, devices=None):
    if devices is not None:
        _run_pipelines_parallel(pipeline_configs, perfsim, devices)
    else:
        _run_pipelines_sequential(pipeline_configs, perfsim)
    #


def _run_pipelines_sequential(pipeline_configs, perfsim=False):
    # get the cwd so that we can continue even if exception occurs
    cwd = os.getcwd()
    for pipeline_config in atpbar.atpbar(pipeline_configs, name='tasks'):
        os.chdir(cwd)
        _run_pipeline_with_log(pipeline_config, perfsim)
    #


def _run_pipelines_parallel(pipeline_configs, perfsim=False, devices=None):
    # get the cwd so that we can continue even if exception occurs
    cwd = os.getcwd()
    num_devices = len(devices) if devices is not None else 0
    parallel_exec = utils.ParallelRun(num_processes=num_devices)
    for model_idx, pipeline_config in enumerate(pipeline_configs):
        os.chdir(cwd)
        device = devices[model_idx % num_devices] if devices is not None else 0
        # pipeline_config = copy.deepcopy(pipeline_config)
        run_pipeline_bound_func = functools.partial(_run_pipeline_with_log, pipeline_config,
                                                    perfsim, device=device)
        parallel_exec.enqueue(run_pipeline_bound_func)
    #
    parallel_exec.start()
    parallel_exec.wait()


def _run_pipeline_with_log(pipeline_config, perfsim=False, device=None):
    if device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    #
    session = pipeline_config['session']
    work_dir = session.get_work_dir()
    os.makedirs(work_dir, exist_ok=True)
    logger = utils.TeeLogger(os.path.join(work_dir, 'run.log'))

    results = None
    try:
        results = run_pipeline(pipeline_config)
    except Exception as e:
        print(f'\n{str(e)}')
    #

    # perfsim_dict = None
    # try:
    #     perfsim_dict = session.perfsim_data() if perfsim else None
    # except Exception as e:
    #     print(f'\n{str(e)}')
    # #
    # if perfsim_dict is not None:
    #     results.update(perfsim_dict)
    # #

    results_message = f'BenchmarkResults: {results}'
    if pipeline_config['verbose_mode']:
        print(results_message)
    else:
        logger.file.write(results_message)
    #
    del logger


def run_pipeline(pipeline_config):
    if pipeline_config['verbose_mode']:
        print('pipeline_config=', pipeline_config)
    #
    pipeline_type = pipeline_config['type']
    if pipeline_type == 'accuracy':
        accuracy_pipeline = AccuracyPipeline()
        results = accuracy_pipeline.run(pipeline_config)
    else:
        assert False, f'unknown pipeline: {pipeline_type}'
    #
    return results


