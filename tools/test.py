# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy
import torch

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.model import is_model_wrapper
from mmengine.logging import print_log
from mmengine.runner import load_checkpoint

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo
import mmdet.hooks
from mmdet.utils.model_optimization import get_replacement_dict, wrap_fn_for_bbox_head, get_input, wrap_fn_for_bbox_head

from edgeai_torchmodelopt import xmodelopt
from edgeai_torchmodelopt import xnn


# TODO: support fuse_conv_bn and format_only
def parse_args(args = None):
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
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
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--model-surgery', type=int, default=None)
    parser.add_argument('--quantization', type=int, default=0)
    args = parser.parse_args(args) if parser else parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main(args=None):
    args = parse_args(args)

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
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

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    cfg.quantization = args.quantization

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))
        
    model_surgery = args.model_surgery
    if args.model_surgery is None:
        if hasattr(cfg,'convert_to_lite_model'):
            model_surgery = cfg.convert_to_lite_model.model_surgery

    if hasattr(cfg, 'resize_with_scale_factor') and cfg.resize_with_scale_factor:
        torch.nn.functional._interpolate_orig = torch.nn.functional.interpolate
        torch.nn.functional.interpolate = xnn.layers.resize_with_scale_factor
    
    # model surgery
    runner._init_model_weights()
    
    is_wrapped = False
    if is_model_wrapper(runner.model):
        runner.model = runner.model.module
        is_wrapped = True

    example_inputs, example_kwargs = get_input(runner.model, cfg,)
    
    transformation_dict = dict(backbone=None, neck=None, bbox_head=xmodelopt.utils.TransformationWrapper(wrap_fn_for_bbox_head))
    copy_attrs=['train_step', 'val_step', 'test_step', 'data_preprocessor', 'parse_losses', 'bbox_head', '_run_forward']
    if model_surgery:
        model_surgery_kwargs = dict(replacement_dict=get_replacement_dict(model_surgery, cfg))
    else:
        model_surgery_kwargs = None
    
    if args.quantization:
        quantization_kwargs = dict(quantization_method='QAT', total_epochs=runner.max_epochs)
    else:
        quantization_kwargs = None
    # if model_surgery_kwargs is not None and quantization_kwargs is None:
    runner.call_hook('before_run')
    runner.load_or_resume()
    runner.call_hook('after_run')
    
    orig_model = deepcopy(runner.model)
    runner.model = xmodelopt.apply_model_optimization(runner.model,example_inputs,example_kwargs, model_surgery_version=model_surgery, quantization_version=args.quantization, model_surgery_kwargs=model_surgery_kwargs, quantization_kwargs=quantization_kwargs, transformation_dict=transformation_dict, copy_attrs=copy_attrs)
    
    if is_wrapped:
        runner.model = runner.wrap_model(
            runner.cfg.get('model_wrapper_cfg'), runner.model)
    print_log('model optimization done')
    # print(runner.model)
    runner.test()


if __name__ == '__main__':
    main()
