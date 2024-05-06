#!/usr/bin/env python


import os
import sys
import datetime
from torch.distributed import launch as distributed_launch
from torchvision.edgeailite import xnn

main_script = './references/detection/train.py'


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
            f'--resize-with-scale-factor',
            f'--pretrained-backbone={args.pretrained_backbone}',
            f'--pretrained={args.pretrained}',
            f'--resume={args.resume}',
            f'--epochs={args.epochs}',
            f'--data-augmentation={args.data_augmentation}',
            f'--lr-scheduler={args.lr_scheduler}'
            ]
    if args.input_size:
        argv += [f'--input-size', f'{args.input_size[0]}', f'{args.input_size[1]}']
    #
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

    args.model = args.model or 'ssdlite_mobilenet_v3_lite_large_fpn'

    args.data_path = args.data_path or './data/datasets/coco'
    args.dataset = args.dataset or 'coco'

    args.mean = [123.675, 116.28, 103.53]
    args.scale = [0.017125, 0.017507, 0.017429]

    args.input_size = [512, 512] #[320, 320]
    args.epochs = 72 #300
    args.lr_scheduler = 'cosineannealinglr'
    args.workers = 4
    args.batch_size = 8
    args.gpus = 4
    args.lr = (0.00078125 * 2 * args.batch_size * args.gpus)
    args.data_augmentation = 'ssd' #'ssd' #'ssdlite' #'mosaic'
    args.weight_decay = 4e-5
    args.tensorboard = True

    args.output_dir = f'./data/checkpoints/detection/{args.dataset}_{args.model}'
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pth')

    if (not args.test_only) and (not args.export_only):
        # training
        is_training = True
        args.pretrained = False
        # args.pretrained_backbone = True # already set as default - can be changed from outside if needed
        args.resume = (checkpoint_path if os.path.exists(checkpoint_path) else '')
        train(args)
    else:
        is_training = False

    if is_training or args.test_only:
        # validation or test
        args.pretrained_backbone = False
        args.pretrained = args.pretrained or checkpoint_path
        test(args)

    if is_training or args.export_only:
        # model export
        args.pretrained_backbone = False
        args.pretrained = args.pretrained or checkpoint_path
        export(args)
