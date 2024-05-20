# Copyright (c) OpenMMLab. All rights reserved.
import copy
#import logging
import os
import os.path as osp
import numpy as np
#import pickle
#import platform
import time
#import warnings
#from collections import OrderedDict
#from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
#from torch.nn.parallel.distributed import DistributedDataParallel
#from torch.optim import Optimizer
from torch.utils.data import DataLoader

import mmengine
from mmengine.config import Config, ConfigDict
#from mmengine.dataset import worker_init_fn as default_worker_init_fn
#from mmengine.device import get_device
from mmengine.dist import broadcast
#from mmengine.dist import (get_dist_info, get_rank, get_world_size,
#                           init_dist, is_distributed, master_only)
from mmengine.evaluator import Evaluator
#from mmengine.fileio import FileClient, join_path
from mmengine.hooks import Hook
#from mmengine.logging import MessageHub, MMLogger, print_log
#from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
#                            is_model_wrapper, revert_sync_batchnorm)
from mmengine.model.efficient_conv_bn_eval import \
    turn_on_efficient_conv_bn_eval
from mmengine.model import is_model_wrapper
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler)
#from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
#                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
#                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
#                               RUNNERS, VISUALIZERS, DefaultScope)
#from mmengine.utils import apply_to, digit_version, get_git_hash, is_seq_of
#from mmengine.utils.dl_utils import (TORCH_VERSION, collect_env,
#                                     set_multi_processing)
from mmengine.visualization import Visualizer
from mmengine.registry import RUNNERS, DefaultScope
from mmengine.runner.activation_checkpointing import \
    turn_on_activation_checkpointing
from mmengine.runner.base_loop import BaseLoop
#from mmengine.checkpoint import (_load_checkpoint, _load_checkpoint_to_model,
#                         find_latest_checkpoint, save_checkpoint,
#                         weights_to_cpu)
#from mmengine.log_processor import LogProcessor
#from mmengine.loops import EpochBasedTrainLoop, IterBasedTrainLoop, TestLoop, ValLoop
#from mmengine.priority import Priority, get_priority
#from mmengine.utils import _get_batch_size, set_random_seed

from torch.nn import functional as F

from mmengine.runner import Runner
from mmengine.runner.checkpoint import ( _load_checkpoint, _load_checkpoint_to_model)

from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape

from .save_onnx import save_onnx_model


ConfigType = Union[Dict, Config, ConfigDict]
ParamSchedulerType = Union[List[_ParamScheduler], Dict[str,
                                                       List[_ParamScheduler]]]
OptimWrapperType = Union[OptimWrapper, OptimWrapperDict]



@RUNNERS.register_module()
class EdgeAIRunner(Runner):
    """A training helper for PyTorch.

    Runner object can be built from config by ``runner = Runner.from_cfg(cfg)``
    where the ``cfg`` usually contains training, validation, and test-related
    configurations to build corresponding components. We usually use the
    same config to launch training, testing, and validation tasks. However,
    only some of these components are necessary at the same time, e.g.,
    testing a model does not need training or validation-related components.

    To avoid repeatedly modifying config, the construction of ``Runner`` adopts
    lazy initialization to only initialize components when they are going to be
    used. Therefore, the model is always initialized at the beginning, and
    training, validation, and, testing related components are only initialized
    when calling ``runner.train()``, ``runner.val()``, and ``runner.test()``,
    respectively.

    Args:
        model (:obj:`torch.nn.Module` or dict): The model to be run. It can be
            a dict used for build a model.
        work_dir (str): The working directory to save checkpoints. The logs
            will be saved in the subdirectory of `work_dir` named
            :attr:`timestamp`.
        train_dataloader (Dataloader or dict, optional): A dataloader object or
            a dict to build a dataloader. If ``None`` is given, it means
            skipping training steps. Defaults to None.
            See :meth:`build_dataloader` for more details.
        val_dataloader (Dataloader or dict, optional): A dataloader object or
            a dict to build a dataloader. If ``None`` is given, it means
            skipping validation steps. Defaults to None.
            See :meth:`build_dataloader` for more details.
        test_dataloader (Dataloader or dict, optional): A dataloader object or
            a dict to build a dataloader. If ``None`` is given, it means
            skipping test steps. Defaults to None.
            See :meth:`build_dataloader` for more details.
        train_cfg (dict, optional): A dict to build a training loop. If it does
            not provide "type" key, it should contain "by_epoch" to decide
            which type of training loop :class:`EpochBasedTrainLoop` or
            :class:`IterBasedTrainLoop` should be used. If ``train_cfg``
            specified, :attr:`train_dataloader` should also be specified.
            Defaults to None. See :meth:`build_train_loop` for more details.
        val_cfg (dict, optional): A dict to build a validation loop. If it does
            not provide "type" key, :class:`ValLoop` will be used by default.
            If ``val_cfg`` specified, :attr:`val_dataloader` should also be
            specified. If ``ValLoop`` is built with `fp16=True``,
            ``runner.val()`` will be performed under fp16 precision.
            Defaults to None. See :meth:`build_val_loop` for more details.
        test_cfg (dict, optional): A dict to build a test loop. If it does
            not provide "type" key, :class:`TestLoop` will be used by default.
            If ``test_cfg`` specified, :attr:`test_dataloader` should also be
            specified. If ``ValLoop`` is built with `fp16=True``,
            ``runner.val()`` will be performed under fp16 precision.
            Defaults to None. See :meth:`build_test_loop` for more details.
        auto_scale_lr (dict, Optional): Config to scale the learning rate
            automatically. It includes ``base_batch_size`` and ``enable``.
            ``base_batch_size`` is the batch size that the optimizer lr is
            based on. ``enable`` is the switch to turn on and off the feature.
        optim_wrapper (OptimWrapper or dict, optional):
            Computing gradient of model parameters. If specified,
            :attr:`train_dataloader` should also be specified. If automatic
            mixed precision or gradient accmulation
            training is required. The type of ``optim_wrapper`` should be
            AmpOptimizerWrapper. See :meth:`build_optim_wrapper` for
            examples. Defaults to None.
        param_scheduler (_ParamScheduler or dict or list, optional):
            Parameter scheduler for updating optimizer parameters. If
            specified, :attr:`optimizer` should also be specified.
            Defaults to None.
            See :meth:`build_param_scheduler` for examples.
        val_evaluator (Evaluator or dict or list, optional): A evaluator object
            used for computing metrics for validation. It can be a dict or a
            list of dict to build a evaluator. If specified,
            :attr:`val_dataloader` should also be specified. Defaults to None.
        test_evaluator (Evaluator or dict or list, optional): A evaluator
            object used for computing metrics for test steps. It can be a dict
            or a list of dict to build a evaluator. If specified,
            :attr:`test_dataloader` should also be specified. Defaults to None.
        default_hooks (dict[str, dict] or dict[str, Hook], optional): Hooks to
            execute default actions like updating model parameters and saving
            checkpoints. Default hooks are ``OptimizerHook``,
            ``IterTimerHook``, ``LoggerHook``, ``ParamSchedulerHook`` and
            ``CheckpointHook``. Defaults to None.
            See :meth:`register_default_hooks` for more details.
        custom_hooks (list[dict] or list[Hook], optional): Hooks to execute
            custom actions like visualizing images processed by pipeline.
            Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`. If the ``model`` argument is a dict
            and doesn't contain the key ``data_preprocessor``, set the argument
            as the ``data_preprocessor`` of the ``model`` dict.
            Defaults to None.
        load_from (str, optional): The checkpoint file to load from.
            Defaults to None.
        resume (bool): Whether to resume training. Defaults to False. If
            ``resume`` is True and ``load_from`` is None, automatically to
            find latest checkpoint from ``work_dir``. If not found, resuming
            does nothing.
        launcher (str): Way to launcher multi-process. Supported launchers
            are 'pytorch', 'mpi', 'slurm' and 'none'. If 'none' is provided,
            non-distributed environment will be launched.
        env_cfg (dict): A dict used for setting environment. Defaults to
            dict(dist_cfg=dict(backend='nccl')).
        log_processor (dict, optional): A processor to format logs. Defaults to
            None.
        log_level (int or str): The log level of MMLogger handlers.
            Defaults to 'INFO'.
        visualizer (Visualizer or dict, optional): A Visualizer object or a
            dict build Visualizer object. Defaults to None. If not
            specified, default config will be used.
        default_scope (str): Used to reset registries location.
            Defaults to "mmengine".
        randomness (dict): Some settings to make the experiment as reproducible
            as possible like seed and deterministic.
            Defaults to ``dict(seed=None)``. If seed is None, a random number
            will be generated and it will be broadcasted to all other processes
            if in distributed environment. If ``cudnn_benchmark`` is
            ``True`` in ``env_cfg`` but ``deterministic`` is ``True`` in
            ``randomness``, the value of ``torch.backends.cudnn.benchmark``
            will be ``False`` finally.
        experiment_name (str, optional): Name of current experiment. If not
            specified, timestamp will be used as ``experiment_name``.
            Defaults to None.
        cfg (dict or Configdict or :obj:`Config`, optional): Full config.
            Defaults to None.

    Note:
        Since PyTorch 2.0.0, you can enable ``torch.compile`` by passing in
        `cfg.compile = True`. If you want to control compile options, you
        can pass a dict, e.g. ``cfg.compile = dict(backend='eager')``.
        Refer to `PyTorch API Documentation <https://pytorch.org/docs/
        master/generated/torch.compile.html#torch.compile>`_ for more valid
        options.

    Examples:
        >>> from mmengine.runner import Runner
        >>> cfg = dict(
        >>>     model=dict(type='ToyModel'),
        >>>     work_dir='path/of/work_dir',
        >>>     train_dataloader=dict(
        >>>     dataset=dict(type='ToyDataset'),
        >>>     sampler=dict(type='DefaultSampler', shuffle=True),
        >>>     batch_size=1,
        >>>     num_workers=0),
        >>>     val_dataloader=dict(
        >>>         dataset=dict(type='ToyDataset'),
        >>>         sampler=dict(type='DefaultSampler', shuffle=False),
        >>>        batch_size=1,
        >>>        num_workers=0),
        >>>     test_dataloader=dict(
        >>>         dataset=dict(type='ToyDataset'),
        >>>         sampler=dict(type='DefaultSampler', shuffle=False),
        >>>         batch_size=1,
        >>>         num_workers=0),
        >>>     auto_scale_lr=dict(base_batch_size=16, enable=False),
        >>>     optim_wrapper=dict(type='OptimizerWrapper', optimizer=dict(
        >>>         type='SGD', lr=0.01)),
        >>>     param_scheduler=dict(type='MultiStepLR', milestones=[1, 2]),
        >>>     val_evaluator=dict(type='ToyEvaluator'),
        >>>     test_evaluator=dict(type='ToyEvaluator'),
        >>>     train_cfg=dict(by_epoch=True, max_epochs=3, val_interval=1),
        >>>     val_cfg=dict(),
        >>>     test_cfg=dict(),
        >>>     custom_hooks=[],
        >>>     default_hooks=dict(
        >>>         timer=dict(type='IterTimerHook'),
        >>>         checkpoint=dict(type='CheckpointHook', interval=1),
        >>>         logger=dict(type='LoggerHook'),
        >>>         optimizer=dict(type='OptimizerHook', grad_clip=False),
        >>>         param_scheduler=dict(type='ParamSchedulerHook')),
        >>>     launcher='none',
        >>>     env_cfg=dict(dist_cfg=dict(backend='nccl')),
        >>>     log_processor=dict(window_size=20),
        >>>     visualizer=dict(type='Visualizer',
        >>>     vis_backends=[dict(type='LocalVisBackend',
        >>>                        save_dir='temp_dir')])
        >>>    )
        >>> runner = Runner.from_cfg(cfg)
        >>> runner.train()
        >>> runner.test()
    """
    cfg: Config
    _train_loop: Optional[Union[BaseLoop, Dict]]
    _val_loop: Optional[Union[BaseLoop, Dict]]
    _test_loop: Optional[Union[BaseLoop, Dict]]

    def __init__(
        self,
        model: Union[nn.Module, Dict],
        work_dir: str,
        train_dataloader: Optional[Union[DataLoader, Dict]] = None,
        val_dataloader: Optional[Union[DataLoader, Dict]] = None,
        test_dataloader: Optional[Union[DataLoader, Dict]] = None,
        train_cfg: Optional[Dict] = None,
        val_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        auto_scale_lr: Optional[Dict] = None,
        optim_wrapper: Optional[Union[OptimWrapper, Dict]] = None,
        param_scheduler: Optional[Union[_ParamScheduler, Dict, List]] = None,
        val_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        test_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
        custom_hooks: Optional[List[Union[Hook, Dict]]] = None,
        data_preprocessor: Union[nn.Module, Dict, None] = None,
        load_from: Optional[str] = None,
        resume: bool = False,
        launcher: str = 'none',
        env_cfg: Dict = dict(dist_cfg=dict(backend='nccl')),
        log_processor: Optional[Dict] = None,
        log_level: str = 'INFO',
        visualizer: Optional[Union[Visualizer, Dict]] = None,
        default_scope: str = 'mmengine',
        randomness: Dict = dict(seed=None),
        experiment_name: Optional[str] = None,
        cfg: Optional[ConfigType] = None,
    ):
        self._work_dir = osp.abspath(work_dir)
        mmengine.mkdir_or_exist(self._work_dir)

        # recursively copy the `cfg` because `self.cfg` will be modified
        # everywhere.
        if cfg is not None:
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

        # lazy initialization
        training_related = [train_dataloader, train_cfg, optim_wrapper]
        if not (all(item is None for item in training_related)
                or all(item is not None for item in training_related)):
            raise ValueError(
                'train_dataloader, train_cfg, and optim_wrapper should be '
                'either all None or not None, but got '
                f'train_dataloader={train_dataloader}, '
                f'train_cfg={train_cfg}, '
                f'optim_wrapper={optim_wrapper}.')
        self._train_dataloader = train_dataloader
        self._train_loop = train_cfg

        self.optim_wrapper: Optional[Union[OptimWrapper, dict]]
        self.optim_wrapper = optim_wrapper

        self.auto_scale_lr = auto_scale_lr

        # If there is no need to adjust learning rate, momentum or other
        # parameters of optimizer, param_scheduler can be None
        if param_scheduler is not None and self.optim_wrapper is None:
            raise ValueError(
                'param_scheduler should be None when optim_wrapper is None, '
                f'but got {param_scheduler}')

        # Parse `param_scheduler` to a list or a dict. If `optim_wrapper` is a
        # `dict` with single optimizer, parsed param_scheduler will be a
        # list of parameter schedulers. If `optim_wrapper` is
        # a `dict` with multiple optimizers, parsed `param_scheduler` will be
        # dict with multiple list of parameter schedulers.
        self._check_scheduler_cfg(param_scheduler)
        self.param_schedulers = param_scheduler

        val_related = [val_dataloader, val_cfg, val_evaluator]
        if not (all(item is None
                    for item in val_related) or all(item is not None
                                                    for item in val_related)):
            raise ValueError(
                'val_dataloader, val_cfg, and val_evaluator should be either '
                'all None or not None, but got '
                f'val_dataloader={val_dataloader}, val_cfg={val_cfg}, '
                f'val_evaluator={val_evaluator}')
        self._val_dataloader = val_dataloader
        self._val_loop = val_cfg
        self._val_evaluator = val_evaluator

        test_related = [test_dataloader, test_cfg, test_evaluator]
        if not (all(item is None for item in test_related)
                or all(item is not None for item in test_related)):
            raise ValueError(
                'test_dataloader, test_cfg, and test_evaluator should be '
                'either all None or not None, but got '
                f'test_dataloader={test_dataloader}, test_cfg={test_cfg}, '
                f'test_evaluator={test_evaluator}')
        self._test_dataloader = test_dataloader
        self._test_loop = test_cfg
        self._test_evaluator = test_evaluator

        self._launcher = launcher
        if self._launcher == 'none':
            self._distributed = False
        else:
            self._distributed = True

        # self._timestamp will be set in the `setup_env` method. Besides,
        # it also will initialize multi-process and (or) distributed
        # environment.
        self.setup_env(env_cfg)
        # self._deterministic and self._seed will be set in the
        # `set_randomness`` method
        self._randomness_cfg = randomness
        self.set_randomness(**randomness)

        if experiment_name is not None:
            self._experiment_name = f'{experiment_name}_{self._timestamp}'
        elif self.cfg.filename is not None:
            filename_no_ext = osp.splitext(osp.basename(self.cfg.filename))[0]
            self._experiment_name = f'{filename_no_ext}_{self._timestamp}'
        else:
            self._experiment_name = self.timestamp
        self._log_dir = osp.join(self.work_dir, self.timestamp)
        mmengine.mkdir_or_exist(self._log_dir)
        # Used to reset registries location. See :meth:`Registry.build` for
        # more details.
        if default_scope is not None:
            default_scope = DefaultScope.get_instance(  # type: ignore
                self._experiment_name,
                scope_name=default_scope)
        self.default_scope = default_scope

        # Build log processor to format message.
        log_processor = dict() if log_processor is None else log_processor
        self.log_processor = self.build_log_processor(log_processor)
        # Since `get_instance` could return any subclass of ManagerMixin. The
        # corresponding attribute needs a type hint.
        self.logger = self.build_logger(log_level=log_level)

        # Collect and log environment information.
        self._log_env(env_cfg)

        # Build `message_hub` for communication among components.
        # `message_hub` can store log scalars (loss, learning rate) and
        # runtime information (iter and epoch). Those components that do not
        # have access to the runner can get iteration or epoch information
        # from `message_hub`. For example, models can get the latest created
        # `message_hub` by
        # `self.message_hub=MessageHub.get_current_instance()` and then get
        # current epoch by `cur_epoch = self.message_hub.get_info('epoch')`.
        # See `MessageHub` and `ManagerMixin` for more details.
        self.message_hub = self.build_message_hub()
        # visualizer used for writing log or visualizing all kinds of data
        self.visualizer = self.build_visualizer(visualizer)
        if self.cfg:
            self.visualizer.add_config(self.cfg)

        self._load_from = load_from
        self._resume = resume
        # flag to mark whether checkpoint has been loaded or resumed
        self._has_loaded = False

        self.voxel_layer = VoxelizationByGridShape(**model['data_preprocessor'].voxel_layer)

        # build a model
        if isinstance(model, dict) and data_preprocessor is not None:
            # Merge the data_preprocessor to model config.
            model.setdefault('data_preprocessor', data_preprocessor)
        self.model = self.build_model(model)

        '''
        if self.cfg.quantize == True:
            from mmdet3d.utils.quantize import XMMDetQuantTrainModule
            import numpy as np

            raw_voxel_feat = np.fromfile("./data/kitti/training/velodyne/000000.bin")
            raw_voxel_feat = torch.tensor(raw_voxel_feat.reshape((-1,4)))

            voxel_dict = self.voxelize([raw_voxel_feat,raw_voxel_feat])

            self.model = XMMDetQuantTrainModule(self.model, voxel_dict, total_epochs=1)
        '''

        # wrap model
        self.model = self.wrap_model(
            self.cfg.get('model_wrapper_cfg'), self.model)

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._hooks: List[Hook] = []
        # register hooks to `self._hooks`
        self.register_hooks(default_hooks, custom_hooks)
        # log hooks information
        self.logger.info(f'Hooks will be executed in the following '
                         f'order:\n{self.get_hooks_info()}')

        # dump `cfg` to `work_dir`
        self.dump_config()

    '''
    def load_checkpoint(self,
                        filename: str,
                        map_location: Union[str, Callable] = 'cpu',
                        strict: bool = False,
                        revise_keys: list = [(r'^module.', '')]):
        """Load checkpoint from given ``filename``.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
            map_location (str or callable): A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'cpu'.
            strict (bool): strict (bool): Whether to allow different params for
                the model and checkpoint.
            revise_keys (list): A list of customized keywords to modify the
                state_dict in checkpoint. Each item is a (pattern, replacement)
                pair of the regular expression operations. Defaults to strip
                the prefix 'module.' by [(r'^module\\.', '')].
        """
        checkpoint = _load_checkpoint(filename, map_location=map_location)

        # Add comments to describe the usage of `after_load_ckpt`
        self.call_hook('after_load_checkpoint', checkpoint=checkpoint)

        #if self.cfg.quantize or is_model_wrapper(self.model):
        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        checkpoint = _load_checkpoint_to_model(
            model, checkpoint, strict, revise_keys=revise_keys)

        self._has_loaded = True

        self.logger.info(f'Load checkpoint from {filename}')

        # Save onnx model here
        #if self.cfg.save_onnx_model is True:
        #    save_onnx_model(self.cfg, model, os.path.dirname(self.cfg.load_from), self.cfg.quantize)

        return checkpoint
    '''

    def test(self) -> dict:
        """Launch test.

        Returns:
            dict: A dict of metrics on testing set.
        """
        if self._test_loop is None:
            raise RuntimeError(
                '`self._test_loop` should not be None when calling test '
                'method. Please provide `test_dataloader`, `test_cfg` and '
                '`test_evaluator` arguments when initializing runner.')

        self._test_loop = self.build_test_loop(self._test_loop)  # type: ignore

        self.call_hook('before_run')

        if self.cfg.quantize == True:
            from mmdet3d.utils.quantize import XMMDetQuantTrainModule
            import numpy as np

            raw_voxel_feat = np.fromfile("./data/kitti/training/velodyne/000000.bin",dtype=np.float32)
            raw_voxel_feat = torch.tensor(raw_voxel_feat.reshape((-1,4)))
            raw_voxel_feat = raw_voxel_feat.cuda()

            voxel_dict = self.voxelize([raw_voxel_feat,raw_voxel_feat])

            self.model = XMMDetQuantTrainModule(self.model, voxel_dict, total_epochs=1)
            # After XMMDetQuantTrainModule, model device is set to cpu
            # So set it to cuda
            self.model = self.model.cuda()

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        import torch.nn.utils.prune as prune

        def parseConv(module, prefix=''):

            if isinstance(module, torch.nn.Conv2d):
                #prune.l1_unstructured(module, name='weight', amount=0.1)
                prune.ln_structured(module, name="weight", amount=0.1, n=float('-inf'), dim=1)
                print("Sparsity in {} {:.2f}%".format(prefix+"Conv2d",
                        100. * float(torch.sum(module.weight == 0))
                        / float(module.weight.nelement())
                    )
                )
                #prune.remove(module, 'weight')

            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.1)
                #prune.ln_structured(module, name="weight", amount=0.1, n=2, dim=3)
                print("Sparsity in {} {:.2f}%".format(prefix+"Linear",
                        100. * float(torch.sum(module.weight == 0))
                        / float(module.weight.nelement())
                    )
                )
                #prune.remove(module, 'weight')

            for name, child in module._modules.items():
                if child is not None:
                    parseConv(child, prefix + name + '.')

        sparsify_model = False
        if sparsify_model is True:
            parseConv(self.model)

        # Save onnx model here
        if self.cfg.quantize or is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        if self.cfg.save_onnx_model is True:
            save_onnx_model(self.cfg, model, os.path.dirname(self.cfg.load_from), self.cfg.quantize)

        metrics = self.test_loop.run()  # type: ignore
        self.call_hook('after_run')
        return metrics

    def _init_model_weights(self) -> None:
        """Initialize the model weights if the model has
        :meth:`init_weights`"""
        model = self.model.module if is_model_wrapper(
            self.model) else self.model

        if hasattr(model, 'init_weights'):
            if self.cfg.quantize == True:
                from mmdet3d.utils.quantize import XMMDetQuantTrainModule

                # sample data file path
                data_dir = os.path.join(self.cfg.train_dataloader.dataset.dataset.data_root, 
                                        self.cfg.train_dataloader.dataset.dataset.data_prefix['pts'])
                sample_bin_file = os.listdir(data_dir)[0]
                raw_voxel_feat = np.fromfile(os.path.join(data_dir,sample_bin_file),dtype=np.float32)
                raw_voxel_feat = torch.tensor(raw_voxel_feat.reshape((-1, self.cfg.train_pipeline[0]['use_dim'])))
                raw_voxel_feat = raw_voxel_feat.cuda()

                voxel_dict = self.voxelize([raw_voxel_feat,raw_voxel_feat])

                model = XMMDetQuantTrainModule(model, voxel_dict, total_epochs=self.cfg.train_cfg.max_epochs)
                # After XMMDetQuantTrainModule, model device is set to cpu
                model = model.cuda()
                model.module.init_weights()

                # sync params and buffers
                for name, params in model.module.state_dict().items():
                    broadcast(params)
            else:
                model.init_weights()

                # sync params and buffers
                for name, params in model.state_dict().items():
                    broadcast(params)


    def train(self) -> nn.Module:
        """Launch training.

        Returns:
            nn.Module: The model after training.
        """
        if is_model_wrapper(self.model):
            ori_model = self.model.module
        else:
            ori_model = self.model
        assert hasattr(ori_model, 'train_step'), (
            'If you want to train your model, please make sure your model '
            'has implemented `train_step`.')

        if self._val_loop is not None:
            assert hasattr(ori_model, 'val_step'), (
                'If you want to validate your model, please make sure your '
                'model has implemented `val_step`.')

        if self._train_loop is None:
            raise RuntimeError(
                '`self._train_loop` should not be None when calling train '
                'method. Please provide `train_dataloader`, `train_cfg`, '
                '`optimizer` and `param_scheduler` arguments when '
                'initializing runner.')

        self._train_loop = self.build_train_loop(
            self._train_loop)  # type: ignore

        # `build_optimizer` should be called before `build_param_scheduler`
        #  because the latter depends on the former
        self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
        # Automatically scaling lr by linear scaling rule
        self.scale_lr(self.optim_wrapper, self.auto_scale_lr)

        if self.param_schedulers is not None:
            self.param_schedulers = self.build_param_scheduler(  # type: ignore
                self.param_schedulers)  # type: ignore

        if self._val_loop is not None:
            self._val_loop = self.build_val_loop(
                self._val_loop)  # type: ignore
        # TODO: add a contextmanager to avoid calling `before_run` many times
        self.call_hook('before_run')

        #seeds are set inside mmdetection. Repeating here to control it properly
        forced_deterministic = True
        # setting --deterministic flag makes training deterministic, even on multiple GPU, multiple worker etc
        # only upsample layer may create difference in run to run. This should be avoided and TrasCOnv instead should be used
        # TIDL model usages upsample layer hence for TIDL model determinisrtic behaviour is not guranteeed.
        # To avoid forgetting setting the flag --deterministic in training, it is set to deterministic by below setting of seeds.
        if forced_deterministic:
            import random
            np.random.seed(self._seed)
            random.seed(self._seed)
            torch.manual_seed(self._seed)
            torch.cuda.manual_seed(self._seed)
            torch.cuda.manual_seed_all(self._seed)
            # When running on the CuDNN backend, two further options must be set
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # initialize the model weights
        self._init_model_weights()

        # try to enable activation_checkpointing feature
        modules = self.cfg.get('activation_checkpointing', None)
        if modules is not None:
            self.logger.info(f'Enabling the "activation_checkpointing" feature'
                             f' for sub-modules: {modules}')
            turn_on_activation_checkpointing(ori_model, modules)

        # try to enable efficient_conv_bn_eval feature
        modules = self.cfg.get('efficient_conv_bn_eval', None)
        if modules is not None:
            self.logger.info(f'Enabling the "efficient_conv_bn_eval" feature'
                             f' for sub-modules: {modules}')
            turn_on_efficient_conv_bn_eval(ori_model, modules)

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        # Initiate inner count of `optim_wrapper`.
        self.optim_wrapper.initialize_count_status(
            self.model,
            self._train_loop.iter,  # type: ignore
            self._train_loop.max_iters)  # type: ignore

        # Maybe compile the model according to options in self.cfg.compile
        # This must be called **AFTER** model has been wrapped.
        self._maybe_compile('train_step')

        model = self.train_loop.run()  # type: ignore
        self.call_hook('after_run')

        if self.cfg.save_onnx_model is True:

            time.sleep(600)

            best_model_dir = os.path.join(self.cfg['work_dir'],'best_KITTI')
            is_best_modal_there = False

            if os.path.exists(best_model_dir):
                best_models = os.listdir(best_model_dir)
                if os.path.exists(os.path.abspath(os.path.join(self.cfg['work_dir'],"best.pth"))):
                    os.unlink(os.path.abspath(os.path.join(self.cfg['work_dir'],"best.pth")))
                os.symlink(os.path.abspath(os.path.join(best_model_dir, best_models[0])), os.path.abspath(os.path.join(self.cfg['work_dir'],"best.pth")))
                is_best_modal_there = True

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            model.to(device)
            model = model.module
            if is_best_modal_there:
                print("converting the model {} into onnx".format(os.path.join(self.cfg['work_dir'], 'best.pth')))
                _load_checkpoint_to_model(model, os.path.join(self.cfg['work_dir'], 'best.pth'))
                save_onnx_model(self.cfg, model, self.cfg['work_dir'], quantized_model=self.cfg['quantize'],tag='best_')
            else:
                print("converting the model {} into onnx".format(os.path.join(self.cfg['work_dir'], 'latest.pth')))
                _load_checkpoint_to_model(model, os.path.join(self.cfg['work_dir'], 'latest.pth'))
                save_onnx_model(self.cfg, model, self.cfg['work_dir'], quantized_model=self.cfg['quantize'],tag='latest_')

        return model


    @torch.no_grad()
    def voxelize(self, points):
        """Apply voxelization to point cloud.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (list[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            Dict[str, Tensor]: Voxelization information.

            - voxels (Tensor): Features of voxels, shape is MxNxC for hard
              voxelization, NxC for dynamic voxelization.
            - coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
              where 1 represents the batch index.
            - num_points (Tensor, optional): Number of points in each voxel.
            - voxel_centers (Tensor, optional): Centers of voxels.
        """
        voxel_dict = dict()
        voxel_dict_sub = dict()

        voxels, coors, num_points, voxel_centers = [], [], [], []
        for i, res in enumerate(points):
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            res_voxel_centers = (
                res_coors[:, [2, 1, 0]] + 0.5) * res_voxels.new_tensor(
                    self.voxel_layer.voxel_size) + res_voxels.new_tensor(
                        self.voxel_layer.point_cloud_range[0:3])
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
            voxel_centers.append(res_voxel_centers)

        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)
        num_points = torch.cat(num_points, dim=0)
        voxel_centers = torch.cat(voxel_centers, dim=0)

        voxel_dict_sub['num_points'] = num_points
        voxel_dict_sub['voxel_centers'] = voxel_centers

        voxel_dict_sub['voxels'] = voxels
        voxel_dict_sub['coors'] = coors

        voxel_dict['voxels'] = voxel_dict_sub

        return voxel_dict

    '''
    def load_or_resume(self) -> None:
        """load or resume checkpoint."""
        if self._has_loaded:
            return None

        # decide to load from checkpoint or resume from checkpoint
        resume_from = None
        if self._resume and self._load_from is None:
            # auto resume from the latest checkpoint
            resume_from = find_latest_checkpoint(self.work_dir)
            self.logger.info(
                f'Auto resumed from the latest checkpoint {resume_from}.')
        elif self._resume and self._load_from is not None:
            # resume from the specified checkpoint
            resume_from = self._load_from

        if resume_from is not None:
            self.resume(resume_from)
            self._has_loaded = True
        elif self._load_from is not None:
            self.load_checkpoint(self._load_from)
            self._has_loaded = True
    '''
