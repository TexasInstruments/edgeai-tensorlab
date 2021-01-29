import os
import functools
import math
import copy

from .. import utils
from . import pipeline_core


def run_pipelines(pipeline_configs, perfsim=True, devices=None):
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
    num_configs = len(pipeline_configs)
    num_sets = math.ceil(num_configs/num_devices)
    for set_idx in range(num_sets):
        parallel_exec = utils.Parallel()
        for device_idx, device in enumerate(devices):
            model_idx = set_idx*num_devices + device_idx
            if model_idx < num_configs:
                os.chdir(cwd)
                pipeline_config = copy.deepcopy(pipeline_configs[model_idx])
                run_pipeline_bound_func = functools.partial(_run_pipeline_with_log, pipeline_config,
                                                                  perfsim, device=device)
                parallel_exec.queue(run_pipeline_bound_func)
            #
        #
        parallel_exec.start()
        parallel_exec.wait()
    #

def _run_pipeline_with_log(pipeline_config, perfsim=True, device=None):
    if device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    #
    model = pipeline_config['model']
    work_dir = model.get_work_dir()
    os.makedirs(work_dir, exist_ok=True)
    logger = utils.TeeLogger(os.path.join(work_dir, 'run.log'))

    results = perfsim_dict = None
    try:
        results, model = run_pipeline(pipeline_config)
    except Exception as e:
        print(f'\n{str(e)}')
    #
    try:
        perfsim_dict = model.perfsim_data() if perfsim else None
    except Exception as e:
        print(f'\n{str(e)}')
    #
    print('\nPerformance Estimate: ', perfsim_dict)
    print('Accuracy Benchmark: ', results)

    if logger is not None:
        logger.close()
    #


def run_pipeline(pipeline_config):
    print(f'\n\n{pipeline_config["inputNetFile"]}')
    print('pipeline_config=', pipeline_config)

    model = pipeline_config['model']
    results = pipeline_core.run(pipeline_config)

    return results, model


