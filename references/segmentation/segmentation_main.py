#!/usr/bin/env python


import os
import sys
import datetime
from torch.distributed import launch as distributed_launch
from torchvision.edgeailite import xnn

main_script = './references/segmentation/train.py'


def get_common_argv(args):
    argv = [f'--data-path={args.data_path}',
            f'--dataset={args.dataset}',
            f'--model={args.model}',
            f'--output-dir={args.output_dir}',
            f'--batch-size={args.batch_size}',
            f'--lr={args.lr}',
            f'--weight-decay={args.weight_decay}',
            f'--workers={args.workers}',
            f'--print-freq={args.print_freq}',
            f'--input-size', f'{args.input_size[0]}', f'{args.input_size[1]}',
            f'--crop-size', f'{args.crop_size[0]}', f'{args.crop_size[1]}',
            f'--resize-with-scale-factor',
            f'--pretrained-backbone={args.pretrained_backbone}',
            f'--pretrained={args.pretrained}',
            f'--resume={args.resume}',
            f'--epochs={args.epochs}',
            f'--data-augmentation={args.data_augmentation}']
    if args.mean:
        argv += [f'--mean', f'{args.mean[0]}', f'{args.mean[1]}', f'{args.mean[2]}']
    #
    if args.scale:
        argv += [f'--scale', f'{args.scale[0]}', f'{args.scale[1]}', f'{args.scale[2]}']
    #
    if args.tensorboard:
        argv += [f'--tensorboard']
    #
    return argv


def train(args):
    sys.argv = [sys.argv[0], f'--nproc_per_node={args.gpus}', '--use_env', main_script] + \
               get_common_argv(args)
    if args.aux_loss:
        sys.argv += [f'--aux-loss']
    #
    distributed_launch.main()


def test(args):
    sys.argv = [sys.argv[0], f'--nproc_per_node={args.gpus}', '--use_env', main_script] + \
               get_common_argv(args) + [f'--test-only']

    distributed_launch.main()


def export(args):
    opset_version = 11

    # hardsigmoid is currently part of torh.onnx - add custom version
    # taken from pytorch/vision pull requests
    def onnx_register_custom_op():
        from torch.onnx.symbolic_helper import parse_args
        @parse_args('v')
        def hardsigmoid(g, self):
            return g.op('HardSigmoid', self, alpha_f=1 / 6)

        from torch.onnx import register_custom_op_symbolic
        register_custom_op_symbolic('::hardsigmoid', hardsigmoid, opset_version)

    onnx_register_custom_op()

    sys.argv = [sys.argv[0]] + get_common_argv(args) + [f'--export-only']

    train_module = xnn.utils.import_file(main_script)
    args = train_module.get_args_parser().parse_args()
    train_module.main(args)


def get_args_parser(add_help=True):
    train_module = xnn.utils.import_file(main_script)
    parser = train_module.get_args_parser(add_help)
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    args.model = args.model or 'deeplabv3plusdws_mobilenet_v3_lite_large'

    args.data_path = args.data_path or './data/datasets/coco'
    args.dataset = args.dataset or 'coco'

    args.gpus = 4
    args.input_size = [512, 512]
    args.crop_size = [480, 480]
    args.mosaic_prob = 0.0
    args.epochs = 60
    args.workers = 4
    args.batch_size = 4
    args.weight_decay = 4e-5
    args.lr = 0.01
    args.tensorboard = True

    args.output_dir = f'./data/checkpoints/segmentation/{args.dataset}_{args.model}'
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth')
    checkpoint_backbone_path = '../edgeai-modelzoo/models/vision/classification/imagenet1k/edgeai-tv/mobilenet_v3_lite_large_20210507_checkpoint.pth'

    if (not args.test_only) and (not args.export_only):
        # training
        is_training = True
        args.aux_loss = False # recommended for 'deeplabv3', 'fcn'
        args.pretrained_backbone = args.pretrained_backbone or checkpoint_backbone_path
        args.resume = (checkpoint_path if os.path.exists(checkpoint_path) else '')
        train(args)
    else:
        is_training = False

    if is_training or args.test_only:
        # validation or test
        args.aux_loss = False
        args.pretrained_backbone = False
        args.pretrained = args.pretrained or checkpoint_path
        test(args)

    if is_training or args.export_only:
        # model export
        args.aux_loss = False
        args.pretrained_backbone = False
        args.pretrained = args.pretrained or checkpoint_path
        export(args)
