# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
import torch
import onnx

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.model import is_model_wrapper
from mmengine.logging import print_log
from mmengine.model.base_module import BaseModule
from mmengine.runner.loops import EpochBasedTrainLoop

from mmdet.utils import setup_cache_size_limit_of_dynamo

from mmdet.utils import convert_to_lite_model
# from mmdet.utils import save_model_proto
from mmdeploy.utils import save_model_proto
from mmdeploy.utils import build_model_from_cfg

from edgeai_torchmodelopt import xmodelopt
from edgeai_torchmodelopt import xnn
from edgeai_torchmodelopt import xonnx

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
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
    parser.add_argument('--export-onnx-model', action='store_true', default=False, help='whether to export the onnx network' )
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main(args=None):

    args = args or parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
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

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

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

    cfg.quantization = args.quantization
    
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
            if args.quantize_calib_images:
                # changing the run_epoch wrapper to run for only calib frames
                del EpochBasedTrainLoop.run_epoch
                
                def run_epoch(self) -> None:
                    """Iterate one epoch."""
                    self.runner.call_hook('before_train_epoch')
                    self.runner.model.train()
                    for idx, data_batch in enumerate(self.dataloader):
                        self.run_iter(idx, data_batch)
                        if idx > args.quantize_calib_images:
                            break
                    self.runner.call_hook('after_train_epoch')
                    self._epoch = self.runner.max_epochs
                    
                setattr(EpochBasedTrainLoop, "run_epoch", run_epoch)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    log_file = runner.logger.log_file
    log_file_link = osp.join(cfg.work_dir, f'run.log')
    xnn.utils.make_symlink(log_file, log_file_link)

    model_surgery = args.model_surgery
    if args.model_surgery is None:
        if hasattr(cfg, 'convert_to_lite_model'):
            model_surgery = cfg.convert_to_lite_model.model_surgery

    if hasattr(cfg, 'resize_with_scale_factor') and cfg.resize_with_scale_factor:
        torch.nn.functional._interpolate_orig = torch.nn.functional.interpolate
        torch.nn.functional.interpolate = xnn.layers.resize_with_scale_factor

    # model surgery
    if model_surgery:
        runner._init_model_weights()

        if model_surgery == 1:
            device = next(runner.model.parameters()).device
            runner.model = convert_to_lite_model(runner.model, cfg)
            runner.model = runner.model.to(torch.device(device))
        elif model_surgery == 2: 
            assert False, 'model surgery 2 is not supported currently'
            surgery_wrapper = xmodelopt.surgery.v2.convert_to_lite_fx

            is_wrapped = False
            if is_model_wrapper(runner.model):
                runner.model = runner.model.module
                is_wrapped = True
            #
                
            if hasattr(runner.model, 'surgery_init'):
                print_log('wrapping the model to prepare for surgery')
                runner.model = runner.model.surgery_init(surgery_wrapper)
            else:
                # raise RuntimeError(f'surgery_init method is not supported for {type(runner.model)}')
                runner.model.backbone = surgery_wrapper(runner.model.backbone)
                # runner.model.neck = surgery_wrapper(runner.model.neck)
                runner.model.bbox_head = surgery_wrapper(runner.model.bbox_head)
            #
                
            if is_wrapped:
                runner.model = runner.wrap_model(runner.cfg.get('model_wrapper_cfg'), runner.model)
        print_log('model surgery done')
        #

    if args.quantization:
        # load the checkpoint before quantization wrapper
        runner.load_or_resume()

        is_wrapped = False
        if is_model_wrapper(runner.model):
            runner.model = runner.model.module
            is_wrapped = True
        #
            
        if args.quantization == xmodelopt.quantization.QuantizationVersion.QUANTIZATION_V1:
            test_loader = runner.build_dataloader(runner._test_dataloader)
            example_input = next(iter(test_loader))
            runner.model = xmodelopt.quantization.v1.QuantTrainModule(
                runner.model, dummy_input=example_input, total_epochs=runner.max_epochs)
        elif args.quantization == xmodelopt.quantization.QuantizationVersion.QUANTIZATION_V2:
            if args.quantize_type in ['PTQ', 'PTC']:
                quantize_wrapper = xmodelopt.quantization.v2.PTCFxModule
            else:
                quantize_wrapper = xmodelopt.quantization.v2.QATFxModule
            if hasattr(runner.model, 'quant_init'):
                print_log('wrapping the model to prepare for quantization')
                runner.model = runner.model.quant_init(
                    quantize_wrapper, total_epochs=runner.max_epochs)
            else:
                print_log(
                    f'quant_init method is not supported for {type(runner.model)}. attempting to use the generic wrapper')
                runner.model = quantize_wrapper(runner.model, total_epochs=runner.max_epochs)
            #
        #
                
        if is_wrapped:
            runner.model = runner.wrap_model(runner.cfg.get('model_wrapper_cfg'), runner.model)
        #
 
    runner.train()

    if args.export_onnx_model or (hasattr(cfg, 'export_onnx_model') and cfg.export_onnx_model):
        # Exporting Model after Training : Uses custom mmdeploy
        try:
            from mmdeploy.apis import torch2onnx
        except:
            raise ModuleNotFoundError
        
        save_file = 'model.onnx'
        save_file = osp.join(cfg.work_dir,save_file)
        
        torch2onnx(img='../edgeai-mmdetection/demo/demo.jpg', work_dir=cfg.work_dir, save_file=save_file, model_cfg = cfg, \
            deploy_cfg='../edgeai-mmdeploy/configs/mmdet/detection/detection_onnxruntime_static.py', torch_model = runner.model)
        
        xonnx.prune_layer_names(save_file, save_file, opset_version=17)
    
        onnx_model = onnx.load(save_file)
        # if args.simplify:
        #     try:
        #         import onnxsim
        #         onnx_model, check = onnxsim.simplify(onnx_model)
        #         assert check, 'assert check failed'
        #     except Exception as e:
        #         print_log(f'Simplify failure: {e}')
        # onnx.save(onnx_model, save_file)
        # print_log(f'ONNX export success, save into {save_file}')

        output_names = ['dets', 'labels']
        feature_names = [node.name for node in onnx_model.graph.output[2:]]
        # write prototxt
        input_shapes = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in onnx_model.graph.input]
        fake_input = torch.randn(*input_shapes[0]).to('cpu')
        save_model_proto(cfg, runner.model, onnx_model, fake_input, save_file, feature_names=feature_names, output_names=output_names)


if __name__ == '__main__':
    main()
