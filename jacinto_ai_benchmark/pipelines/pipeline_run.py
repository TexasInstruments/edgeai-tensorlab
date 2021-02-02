import os
import functools
import math
import copy

from .. import utils
from . import accuracy


def run(pipeline_configs, perfsim=True, devices=None):
    if devices is not None:
        _run_pipelines_parallel(pipeline_configs, perfsim, devices)
    else:
        _run_pipelines_sequential(pipeline_configs, perfsim)
    #


def _run_pipelines_sequential(pipeline_configs, perfsim=True):
    # get the cwd so that we can continue even if exception occurs
    cwd = os.getcwd()
    for pipeline_config in pipeline_configs:
        os.chdir(cwd)
        _run_pipeline_with_log(pipeline_config, perfsim)
    #


def _run_pipelines_parallel(pipeline_configs, perfsim=True, devices=None):
    # get the cwd so that we can continue even if exception occurs
    cwd = os.getcwd()
    num_devices = len(devices)
    parallel_exec = utils.Parallel(len(devices))
    for model_idx, pipeline_config in enumerate(pipeline_configs):
        os.chdir(cwd)
        device = devices[model_idx % num_devices]
        # pipeline_config = copy.deepcopy(pipeline_config)
        run_pipeline_bound_func = functools.partial(_run_pipeline_with_log, pipeline_config,
                                                    perfsim, device=device)
        parallel_exec.enqueue(run_pipeline_bound_func)
    #
    parallel_exec.start()
    parallel_exec.wait()

def _run_pipeline_with_log(pipeline_config, perfsim=True, device=None):
    if device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    #
    session = pipeline_config['session']
    work_dir = session.get_work_dir()
    os.makedirs(work_dir, exist_ok=True)
    logger = utils.TeeLogger(os.path.join(work_dir, 'run.log'))

    results = perfsim_dict = None
    try:
        results = run_pipeline(pipeline_config)
    except Exception as e:
        print(f'\n{str(e)}')
    #
    try:
        perfsim_dict = session.perfsim_data() if perfsim else None
    except Exception as e:
        print(f'\n{str(e)}')
    #
    print('\nPerformance Estimate: ', perfsim_dict)
    print('Accuracy Benchmark: ', results)

    if logger is not None:
        logger.close()
    #


def run_pipeline(pipeline_config):
    print('pipeline_config=', pipeline_config)
    pipeline_type = pipeline_config['type']
    if pipeline_type == 'accuracy':
        results = accuracy.run(pipeline_config)
    else:
        assert False, f'unknown pipeline: {pipeline_type}'
    #
    return results


