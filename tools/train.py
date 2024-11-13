# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmengine.model import is_model_wrapper
from mmengine.model.base_module import BaseModule
from mmengine.model.base_model import BaseModel
from mmengine.runner.loops import EpochBasedTrainLoop, ValLoop
from mmengine.utils.path import symlink

from mmdet3d.utils import replace_ceph_backend

from mmdet3d.utils.model_optimization import get_replacement_dict, wrap_optimize_func, get_input, wrap_fn_for_bbox_head
from mmengine.device import get_device

import numpy as np
import torch
from edgeai_torchmodelopt import xmodelopt

import warnings
from copy import deepcopy

def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--sync_bn',
        choices=['none', 'torch', 'mmcv'],
        default='none',
        help='convert all BatchNorm layers in the model to SyncBatchNorm '
        '(SyncBN) or mmcv.ops.sync_bn.SyncBatchNorm (MMSyncBN) layers.')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--model-surgery', type=int, default=None)
    parser.add_argument('--quantization', type=int, default=0)
    parser.add_argument('--quantize-type', type=str, default='QAT')
    parser.add_argument('--quantize-calib-images', type=int, default=50)
    parser.add_argument('--max-eval-samples', type=int, default=2000)
    parser.add_argument('--export-onnx-model', action='store_true', default=False, help='whether to export the onnx network' )
    parser.add_argument('--simplify', action='store_true', default=False, help='whether to simplify the onnx model or not model' )   
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg["device"] = get_device()

    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # convert BatchNorm layers
    if args.sync_bn != 'none':
        cfg.sync_bn = args.sync_bn

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    '''
    #seeds are set inside mmdetection. Repeating here to control it properly
    forced_deterministic = True
    # setting --deterministic flag makes training deterministic, even on multiple GPU, multiple worker etc
    # only upsample layer may create difference in run to run. This should be avoided and TrasCOnv instead should be used
    # TIDL model usages upsample layer hence for TIDL model determinisrtic behaviour is not guranteeed.
    # To avoid forgetting setting the flag --deterministic in training, it is set to deterministic by below setting of seeds.
    if forced_deterministic:
        import random
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
    '''

    if hasattr(cfg,'save_onnx_model') is False:
        cfg.save_onnx_model = False

    if hasattr(cfg,'quantize') is False:
        cfg.quantize = False
        
    if args.quantization:
        if 'custom_hooks' in cfg:
                hooks_to_remove = ['EMAHook']
                for hook_type in hooks_to_remove:
                    if any([hook_cfg.type == hook_type for hook_cfg in cfg.custom_hooks]):
                        warnings.warn(f'{hook_type} is currently not supported in quantization - removing it')
                    #
                    cfg.custom_hooks = [hook_cfg for hook_cfg in cfg.custom_hooks if hook_cfg.type != hook_type]
                #
        else:
            cfg.custom_hooks = []
        
         # remove the init_weights wrapper from model, else train always calls init_weights wrapper
        del BaseModule.init_weights
        
        if args.quantize_type in ['PTQ', 'PTC']:

            del BaseModel.train_step

            def train_step(self, data, optim_wrapper):
                """
                Calls ``optim_wrapper.update_params(loss)`` to update model has been removed from orig model for PTQ
                """
                # Enable automatic mixed precision training context.
                with optim_wrapper.optim_context(self):
                    data = self.data_preprocessor(data, True)
                    losses = self._run_forward(data, mode='loss')  # type: ignore
                parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
                # optim_wrapper.update_params(parsed_losses)
                return log_vars

            setattr(BaseModel, "train_step", train_step)


            if args.quantize_calib_images:
                # changing the run_epoch wrapper to run for only calib frames
                del EpochBasedTrainLoop.run_epoch
                
                def run_epoch(self) -> None:
                    """Iterate one epoch."""
                    self.runner.call_hook('before_train_epoch')
                    self.runner.model.train()
                    for idx, data_batch in enumerate(self.dataloader):
                        self.run_iter(idx, data_batch)
                        if idx*len(data_batch) > args.quantize_calib_images:
                            break
                    self.runner.call_hook('after_train_epoch')
                    self._epoch = self.runner.max_epochs
                    
                setattr(EpochBasedTrainLoop, "run_epoch", run_epoch)


    # del EpochBasedTrainLoop.run_epoch
                
    # def run_epoch(self) -> None:
    #     """Iterate one epoch."""
    #     self.runner.call_hook('before_train_epoch')
    #     self.runner.model.train()
    #     for idx, data_batch in enumerate(self.dataloader):
    #         self.run_iter(idx, data_batch)
    #         if idx*len(data_batch) > 1:
    #             break
    #     self.runner.call_hook('after_train_epoch')
    #     self._epoch = self.runner.max_epochs
        
    # setattr(EpochBasedTrainLoop, "run_epoch", run_epoch)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)


    # log_file = runner.logger.log_file
    # log_file_link = osp.join(cfg.work_dir, f'run.log')
    # xnn.utils.make_symlink(log_file, log_file_link)

    model_surgery = args.model_surgery
    if args.model_surgery is None:
        if hasattr(cfg, 'convert_to_lite_model'):
            model_surgery = cfg.convert_to_lite_model.model_surgery
        else:
            model_surgery = 0

    # if hasattr(cfg, 'resize_with_scale_factor') and cfg.resize_with_scale_factor:
    #     torch.nn.functional._interpolate_orig = torch.nn.functional.interpolate
    #     torch.nn.functional.interpolate = xnn.layers.resize_with_scale_factor

    # model surgery
    runner._init_model_weights()
    
    is_wrapped = False
    if is_model_wrapper(runner.model):
        runner.model = runner.model.module
        is_wrapped = True

    example_inputs, example_kwargs = get_input(runner.model, cfg,)
    
    transformation_dict = dict(backbone=None, neck=None, bbox_head=xmodelopt.TransformationWrapper(wrap_fn_for_bbox_head))
    copy_attrs=['train_step', 'val_step', 'test_step', 'data_preprocessor', 'parse_losses', 'bbox_head', '_run_forward']
    if model_surgery:
        model_surgery_kwargs = dict(replacement_dict=get_replacement_dict(model_surgery, cfg))
    else:
        model_surgery_kwargs = None
    
    if args.quantization:
        quantization_kwargs = dict(quantization_method=args.quantize_type, total_epochs=runner.max_epochs)
    else:
        quantization_kwargs = None
    # if model_surgery_kwargs is not None and quantization_kwargs is None:
    runner.call_hook('before_run')
    runner.load_or_resume()
    runner.call_hook('after_run')
    
    orig_model = deepcopy(runner.model)
    runner.model = xmodelopt.apply_model_optimization(runner.model, example_inputs, example_kwargs, model_surgery_version=model_surgery, 
                                                      quantization_version=args.quantization, model_surgery_kwargs=model_surgery_kwargs, 
                                                      quantization_kwargs=quantization_kwargs, transformation_dict=transformation_dict, 
                                                      copy_attrs=copy_attrs)
    
    if is_wrapped:
        runner.model = runner.wrap_model(
            runner.cfg.get('model_wrapper_cfg'), runner.model)
    print_log('model optimization done')

    # start training
    runner.train()
    
    # if xnn.utils.distributed_utils.is_main_process() and (args.export_onnx_model or (hasattr(cfg, 'export_onnx_model') and cfg.export_onnx_model)):
    #     # Exporting Model after Training : Uses custom mmdeploy
    #     try:
    #         from mmdeploy.apis import torch2onnx
    #     except:
    #         raise ModuleNotFoundError
        
    #     save_file = 'model.onnx'
    #     save_file = osp.join(cfg.work_dir,save_file)
    #     is_wrapped = False
    #     if is_model_wrapper(runner.model):
    #         runner.model = runner.model.module
    #         is_wrapped = True
    #     example_inputs, example_kwargs = get_input(runner.model, cfg, batch_size=1, to_export=True)
    #     runner.model = xmodelopt.prepare_model_for_onnx(orig_model, runner.model, example_inputs, example_kwargs, model_surgery_version=model_surgery, 
    #                                                     quantization_version=args.quantization, model_surgery_kwargs=model_surgery_kwargs, 
    #                                                     quantization_kwargs=quantization_kwargs, transformation_dict=transformation_dict, copy_attrs=copy_attrs)
        
        
    #     torch2onnx(img='../edgeai-mmdetection/demo/demo.jpg', work_dir=cfg.work_dir, save_file=save_file, model_cfg = cfg, \
    #         deploy_cfg='../edgeai-mmdeploy/configs/mmdet/detection/detection_onnxruntime_static.py', torch_model = runner.model)
        
    #     xonnx.prune_layer_names(save_file, save_file, opset_version=17)
    
    #     onnx_model = onnx.load(save_file)
    #     if args.simplify:
    #         try:
    #             import onnxsim
    #             onnx_model, check = onnxsim.simplify(onnx_model)
    #             assert check, 'assert check failed'
    #         except Exception as e:
    #             print_log(f'Simplify failure: {e}')
    #     onnx.save(onnx_model, save_file)
    #     print_log(f'ONNX export success, save into {save_file}')

    #     output_names = ['dets', 'labels']
    #     feature_names = [node.name for node in onnx_model.graph.output[2:]]
    #     # write prototxt
    #     input_shapes = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in onnx_model.graph.input]
    #     fake_input = torch.randn(*input_shapes[0]).to('cpu')
    #     save_model_proto(cfg, getattr(runner.model,'module',runner.model), onnx_model, fake_input, save_file, feature_names=feature_names, output_names=output_names)
    # else:
    #     return


if __name__ == '__main__':
    main()
