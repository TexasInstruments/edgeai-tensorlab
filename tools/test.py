# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.model import is_model_wrapper
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.model.base_module import BaseModule
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend

from mmdet3d.utils.model_optimization import get_replacement_dict, get_input, wrap_fn_for_bbox_head, replace_dform_conv_with_split_offset_mask, modify_runner_load_check_point_function
from mmengine.device import get_device

import numpy as np
import torch
from edgeai_torchmodelopt import xmodelopt

import warnings
from copy import deepcopy


# TODO: support fuse_conv_bn and format_only
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--task',
        type=str,
        choices=[
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--model-surgery', type=int, default=None)
    parser.add_argument('--quantization', type=int, default=0)
    parser.add_argument('--quantize-type', type=str, default='QAT')
    parser.add_argument('--quantize-calib-images', type=int, default=50)
    parser.add_argument('--max-eval-samples', type=int, default=2000)
    parser.add_argument('--export-onnx-model', action='store_true', default=False, help='whether to export the onnx network' )
    parser.add_argument('--simplify', action='store_true', default=False, help='whether to simplify the onnx model or not model' )   
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/test.py` instead
    # of `--local_rank`.
    # parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args() if args is None else parser.parse_args(args)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
        all_task_choices = [
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ]
        assert args.task in all_task_choices, 'You must set '\
            f"'--task' in {all_task_choices} in the command " \
            'if you want to use visualization hook'
        visualization_hook['vis_task'] = args.task
        visualization_hook['score_thr'] = args.score_thr
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main(args=None):
    args = parse_args(args) 

    # load config
    cfg = Config.fromfile(args.config)

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

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        # Currently, we only support tta for 3D segmentation
        # TODO: Support tta for 3D detection
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.'
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` in config.'
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)

    if hasattr(cfg,'save_onnx_model') is False:
        cfg.save_onnx_model = False

    if hasattr(cfg,'quantize') is False:
        cfg.quantize = False

    if hasattr(cfg,'match_tidl_nms') is False:
        cfg.match_tidl_nms = True

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # Need to validate model optimization for other models
    if args.quantization and \
       (cfg.get("model")['type'] == 'FCOSMono3D' or \
        cfg.get("model")['type'] == 'FastBEV' or \
        cfg.get("model")['type'] == 'PETR'):

        model_surgery = args.model_surgery
        if args.model_surgery is None:
            if hasattr(cfg, 'convert_to_lite_model'):
                model_surgery = cfg.convert_to_lite_model.model_surgery
            else:
                model_surgery = 0

        # if hasattr(cfg, 'resize_with_scale_factor') and cfg.resize_with_scale_factor:
        #     torch.nn.functional._interpolate_orig = torch.nn.functional.interpolate
        #     torch.nn.functional.interpolate = xnn.layers.resize_with_scale_factor

        is_wrapped = False
        if is_model_wrapper(runner.model):
            runner.model = runner.model.module
            is_wrapped = True

        example_inputs, example_kwargs = get_input(runner, cfg, train=False)

        # can we one unified transfomration_dict for all models?
        if cfg.get("model")['type'] == 'FCOSMono3D':
            transformation_dict = dict(backbone=None, neck=None, bbox_head=xmodelopt.utils.TransformationWrapper(wrap_fn_for_bbox_head))
        elif cfg.get("model")['type'] == 'FastBEV':
            transformation_dict = dict(backbone=None, neck=None, neck_fuse_0=None, neck_3d=None, bbox_head=xmodelopt.utils.TransformationWrapper(wrap_fn_for_bbox_head))
        elif cfg.get("model")['type'] == 'PETR':
            transformation_dict = dict(img_neck=None, img_backbone=None, grid_mask=None, pts_bbox_head=xmodelopt.utils.TransformationWrapper(wrap_fn_for_bbox_head))
        else:
            raise RuntimeError('Quantization is NOT supported for this model')


        copy_attrs=['train_step', 'val_step', 'test_step', 'data_preprocessor', 'parse_losses', 'bbox_head', '_run_forward']

        if model_surgery:
            model_surgery_kwargs = dict(replacement_dict=get_replacement_dict(model_surgery, cfg))
        else:
            model_surgery_kwargs = None

        if args.quantization:
            if args.quantize_type in ['PTQ', 'PTC']:
                quantization_kwargs = dict(quantization_method=args.quantize_type, total_epochs=2, qconfig_type="WC8_AT8")
            else:
                quantization_kwargs = dict(quantization_method=args.quantize_type, total_epochs=runner.max_epochs, qconfig_type="WC8_AT8")
        else:
            quantization_kwargs = None

        # if model_surgery_kwargs is not None and quantization_kwargs is None:
        orig_model = deepcopy(runner.model)
        runner.model = xmodelopt.apply_model_optimization(runner.model, example_inputs, example_kwargs, model_surgery_version=model_surgery, 
                                                            quantization_version=args.quantization, model_surgery_kwargs=model_surgery_kwargs, 
                                                            quantization_kwargs=quantization_kwargs, transformation_dict=transformation_dict, 
                                                            copy_attrs=copy_attrs)

        if is_wrapped:
            runner.model = runner.wrap_model(
                runner.cfg.get('model_wrapper_cfg'), runner.model)

        # runner._init_model_weights()
        # start testing
        # runner._init_model_weights()
        # del BaseModule.init_weights

        runner.model.eval()
        # runner.call_hook('before_run')
        modify_runner_load_check_point_function(runner)
        runner.load_or_resume()
        # runner.call_hook('after_run')
        # runner.model = replace_dform_conv_with_split_offset_mask(runner.model)

    runner.test()


if __name__ == '__main__':
    main()
