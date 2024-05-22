# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

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

from edgeai_torchmodelopt import xonnx

from edgeai_torchmodelopt import xonnx


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
    parser.add_argument('--model-surgery', type=int, default=0)
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify onnx model by onnx-sim')
    parser.add_argument('--model-surgery', type=int, default=0)
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
    

    if args.img_size :
        fake_input = torch.randn(args.batch_size, 3,
                             *args.img_size).to(args.device)
        torch2onnx(
            fake_input,
            args.work_dir,
            save_file,
            deploy_cfg=args.deploy_cfg,
            model_cfg=args.model_cfg,
            model_checkpoint=args.checkpoint,
            device=args.device,
            model_surgery=args.model_surgery)
    else :
        torch2onnx(
            args.img,
            args.work_dir,
            save_file,
            deploy_cfg=args.deploy_cfg,
            model_cfg=args.model_cfg,
            model_checkpoint=args.checkpoint,
            device=args.device,
            model_surgery=args.model_surgery)

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
