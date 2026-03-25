import numpy as np
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.dist.utils import master_only
from mmdet3d.registry import HOOKS 

import os
import warnings
import datetime
import torch

try:
    import mlflow
    has_mlflow = True
except ImportError:
    has_mlflow = False
    warnings.warn('Please install mlflow and login with your api key to log the results to mlflow')


def add_steps(func):
    @master_only
    def wrapper(self, runner: Runner, *args, **kwargs):
        step = self.steps.get(func.__name__,0)
        result = func(self, runner, *args, **kwargs)
        self.steps[func.__name__] = step+1
        return result
    return wrapper


if has_mlflow:
    @HOOKS.register_module()
    class MlflowHook(Hook):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.run_started = False
            self.steps = {}
            self.server_on = True

        @master_only
        def create_mlflow_session(self, runner: Runner):
            file_name = os.path.splitext(os.path.basename(runner.cfg.filename))[0]
            if (expt:= mlflow.get_experiment_by_name('mmdet3d_'+file_name)):
                self.expt_id = expt.experiment_id
            else:
                self.expt_id = mlflow.create_experiment('mmdet3d_'+file_name, artifact_location=os.path.join(runner.work_dir,'mlruns'))
            self.experiment = mlflow.set_experiment(experiment_id=self.expt_id)
            # if not mlflow.is_tracking_uri_set():
            # mlflow.set_tracking_uri('http://127.0.0.1:5000')
            mlflow.start_run(run_name= f'{datetime.datetime.now():%Y_%m_%d_%H_%M_%S}:', experiment_id=self.expt_id)
            self.run = mlflow.active_run() 
            runner.logger.info(f'Run ID: {self.run.info.run_id} started with status {self.run.info.status}')
            
        @master_only
        def before_run(self, runner:Runner):
            if not self.server_on:
                pass
            if not self.run_started:
                self.create_mlflow_session(runner)
                self.run_started = True
            mlflow.log_params(runner.cfg)
            return super().before_run(runner)
        
        @master_only
        def after_run(self, runner: Runner):
            if self.run_started:
                mlflow.end_run()
                runner.logger.info(f'Run ID: {self.run.info.run_id} ended with status {self.run.info.status}')
                self.run_started = False
            return super().after_run(runner)
        @add_steps
        def after_train_iter(self, runner, batch_idx, data_batch = None, outputs = None):
            if 'after_train_iter' not in self.steps:
                self.steps['after_train_iter'] = 0
            if self.run_started:
                for key, val in outputs.items():
                    if isinstance(val, torch.Tensor):
                        val = val.cpu()
                        val = val.detach().numpy().tolist()
                    mlflow.log_metric(key, val, step=self.steps['after_train_iter'])
            return getattr(super(), 'after_train_iter')( runner, batch_idx=batch_idx, data_batch=data_batch, outputs=outputs)
        
        @add_steps
        def after_test_epoch(self, runner, metrics = None):
            if 'after_test_epoch' not in self.steps:
                self.steps['after_test_epoch'] = 0
            if self.run_started:
                for key, val in metrics.items():
                    mlflow.log_metric(key, val, step=self.steps['after_test_epoch'])
            return getattr(super(), 'after_test_epoch')(runner, metrics=metrics)
        
        @add_steps
        def after_val_epoch(self, runner, metrics = None):
            if 'after_val_epoch' not in self.steps:
                self.steps['after_val_epoch'] = 0
            if self.run_started:
                for key, val in metrics.items():
                    mlflow.log_metric(key, val, step=self.steps['after_val_epoch'])
            return  getattr(super(), 'after_val_epoch')(runner, metrics=metrics)

else:
    @HOOKS.register_module()
    class MlflowHook(Hook):
        '''
        Just to have a Hook to register it in the registry which does nothing
        '''
        def __init__(self, *args, **kwargs):
            super().__init__()
            

