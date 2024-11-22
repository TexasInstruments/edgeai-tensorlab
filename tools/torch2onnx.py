# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
from copy import deepcopy

import torch
import onnx
# from mmdet.utils import save_model_proto
from mmdeploy.utils import save_model_proto
from mmdeploy.apis import (extract_model, get_predefined_partition_cfg,
                           torch2onnx)
from mmdeploy.utils import (get_ir_config, get_partition_config,
                            get_root_logger, load_config)
from mmdeploy.utils import build_model_from_cfg
from mmengine.logging import print_log
from mmengine.runner import load_checkpoint
from mmdet.utils.model_optimization import get_replacement_dict, wrap_fn_for_bbox_head, get_input

from edgeai_torchmodelopt import xonnx
from edgeai_torchmodelopt import xnn
from edgeai_torchmodelopt import xmodelopt

def parse_args():
    parser = argparse.ArgumentParser(description='Export model to ONNX.')
    parser.add_argument('deploy_cfg', help='deploy config path')
    parser.add_argument('model_cfg', help='model config path')
    parser.add_argument('checkpoint', help='model checkpoint path')
    parser.add_argument('img', help='image used to convert model model')
    parser.add_argument('--batch-size', help='image size used to convert model model', default=1)
    parser.add_argument(
        '--img-size',
        nargs='+',
        type=int,
        default=None,
        help='Image size of height and width')
    parser.add_argument(
        '--work-dir',
        default='./work-dir',
        help='Directory to save output files.')
    parser.add_argument(
        '--device', help='device used for conversion', default='cpu')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify onnx model by onnx-sim')
    parser.add_argument('--model-surgery', type=int, default=None)
    parser.add_argument(
        '--keep-layer-names',
        action='store_true',
        help='do not rename onnx layers for TIDL')
    parser.add_argument(
        '--log-level',
        help='set log level',
        default='INFO',
        choices=list(logging._nameToLevel.keys()))
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    logger = get_root_logger(log_level=args.log_level)

    logger.info(f'torch2onnx: \n\tmodel_cfg: {args.model_cfg} '
                f'\n\tdeploy_cfg: {args.deploy_cfg}')

    os.makedirs(args.work_dir, exist_ok=True)
    # load deploy_cfg
    deploy_cfg = load_config(args.deploy_cfg)[0]
    model_cfg = load_config(args.model_cfg)[0]
    # save_file = get_ir_config(deploy_cfg)['save_file']

    save_file = osp.join(
        args.work_dir,
        osp.basename(args.model_cfg).replace('py', 'onnx'))

    if hasattr(model_cfg, 'resize_with_scale_factor') and model_cfg.resize_with_scale_factor:
        torch.nn.functional._interpolate_orig = torch.nn.functional.interpolate
        torch.nn.functional.interpolate = xnn.layers.resize_with_scale_factor

    # torch_model = task_processor.build_pytorch_model(model_checkpoint)
    torch_model = build_model_from_cfg(args.model_cfg, args.checkpoint, args.device)

    #model surgery
    model_surgery = args.model_surgery
    if args.model_surgery is None:
        if hasattr(model_cfg, 'convert_to_lite_model'):
            model_surgery = model_cfg.convert_to_lite_model.model_surgery
    # model surgery
    img_size = args.img_size

    example_inputs, example_kwargs = get_input(torch_model, model_cfg, batch_size=args.batch_size, img_size=img_size) 
    
    transformation_dict = dict(backbone=None, neck=None, bbox_head=xmodelopt.utils.TransformationWrapper(wrap_fn_for_bbox_head))
    copy_attrs=['train_step', 'val_step', 'test_step', 'data_preprocessor', 'parse_losses', 'bbox_head', '_run_forward']
    if model_surgery:
        model_surgery_kwargs = dict(replacement_dict=get_replacement_dict(model_surgery, model_cfg))
    else:
        model_surgery_kwargs = None

    torch_model = xmodelopt.apply_model_optimization(torch_model,example_inputs,example_kwargs, model_surgery_version=model_surgery, quantization_version=args.quantization, model_surgery_kwargs=model_surgery_kwargs, transformation_dict=transformation_dict, copy_attrs=copy_attrs)

    print_log('model optimization done')
    
    load_checkpoint(torch_model, args.checkpoint, map_location='cpu')

    fake_input = example_inputs[0]
    if img_size :
        fake_input = fake_input.to(args.device)
        torch2onnx(
            fake_input,
            args.work_dir,
            save_file,
            deploy_cfg=args.deploy_cfg,
            model_cfg=args.model_cfg,
            model_checkpoint=args.checkpoint,
            device=args.device,
            torch_model=torch_model)
    else :
        torch2onnx(
            args.img,
            args.work_dir,
            save_file,
            deploy_cfg=args.deploy_cfg,
            model_cfg=args.model_cfg,
            model_checkpoint=args.checkpoint,
            device=args.device,
            torch_model=torch_model)

    # partition model
    partition_cfgs = get_partition_config(deploy_cfg)

    if partition_cfgs is not None:
        if 'partition_cfg' in partition_cfgs:
            partition_cfgs = partition_cfgs.get('partition_cfg', None)
        else:
            assert 'type' in partition_cfgs
            partition_cfgs = get_predefined_partition_cfg(
                deploy_cfg, partition_cfgs['type'])

        origin_ir_file = osp.join(args.work_dir, save_file)
        for partition_cfg in partition_cfgs:
            save_file = partition_cfg['save_file']
            save_path = osp.join(args.work_dir, save_file)
            start = partition_cfg['start']
            end = partition_cfg['end']
            dynamic_axes = partition_cfg.get('dynamic_axes', None)

            extract_model(
                origin_ir_file,
                start,
                end,
                dynamic_axes=dynamic_axes,
                save_file=save_path)
    logger.info(f'torch2onnx finished. Results saved to {args.work_dir}')

    output_prefix = osp.join(args.work_dir,
                             osp.splitext(osp.basename(save_file))[0])
    # save_onnx_path = output_prefix + '.onnx'
    # check the layers names and shorten it required.
    if not args.keep_layer_names:
        xonnx.prune_layer_names(save_file, save_file, opset_version=17)
    
    onnx_model = onnx.load(save_file)
    if args.simplify:
        try:
            import onnxsim
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print_log(f'Simplify failure: {e}')
    onnx.save(onnx_model, save_file)
    print_log(f'ONNX export success, save into {save_file}')

    model = build_model_from_cfg(args.model_cfg, args.checkpoint,
                                 args.device)

    output_names = ['dets', 'labels']
    feature_names = [node.name for node in onnx_model.graph.output[2:]]
    # write prototxt
    if not args.img_size:
        input_shapes = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in onnx_model.graph.input]
        fake_input = torch.randn(*input_shapes[0]).to(args.device)
    save_model_proto(model_cfg, model, onnx_model, fake_input, save_file, feature_names=feature_names, output_names=output_names)


if __name__ == '__main__':
    main()
