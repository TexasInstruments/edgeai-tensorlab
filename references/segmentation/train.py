import datetime
import os
import time
from colorama import Fore
import numpy as np
import warnings

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchinfo

from torchvision.edgeailite import xnn

from coco_utils import get_coco
import presets
import utils
import onnx

def get_dataset(dir_path, name, image_set, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)

    paths = {
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco": (dir_path, get_coco, 21),
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(args, train):
    base_size = args.input_size
    crop_size = args.crop_size

    return presets.SegmentationPresetTrain(base_size, crop_size, image_mean=args.image_mean, image_scale=args.image_scale, data_augmentation=args.data_augmentation) if train else presets.SegmentationPresetEval(base_size, image_mean=args.image_mean, image_scale=args.image_scale)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def round_frac(number, digits=4):
    return round(number, digits)


def tensor_to_numpy(tensor, image_mean=None, image_scale=None, dataformats=None):
    if dataformats is None:
        dataformats = 'HW' if tensor.ndim == 2 else ('CHW' if tensor.ndim == 3 else dataformats)
    #
    assert tensor.ndim in (2,3,4), 'tensor_to_numpy supports only 2, 3 or 4 dim tensors'
    array = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
    if array.ndim > 2:
        if dataformats == 'HWC':
            view_args = (1,1,-1) if tensor.ndim == 3 else (1,1,1,-1)
        elif dataformats == 'CHW':
            view_args = (-1,1,1) if tensor.ndim == 3 else (1,-1,1,1)
        else:
            view_args = (-1,)
        #
    #
    array = array / image_scale.reshape(view_args) if image_scale is not None else array
    array = array + image_mean.reshape(view_args) if image_scale is not None else array
    return array


def write_visualization(visualizer, name, tensor, epoch, image_mean=None, image_scale=None, image_formatting=None, dataformats=None):
    if dataformats is None:
        dataformats = 'HW' if tensor.ndim == 2 else ('CHW' if tensor.ndim == 3 else dataformats)
    #
    tensor = tensor_to_numpy(tensor, image_mean=image_mean, image_scale=image_scale, dataformats=dataformats)
    if image_formatting is not None:
        tensor = np.array(tensor.clip(0, 255.0), dtype=np.uint8)
    #
    visualizer.add_image(name, tensor, epoch, dataformats=dataformats)


def tensor_to_color(seg_image, num_classes, dataformats='CHW'):
    assert seg_image.ndim <= 3, 'supports only tensor of dimension 2 or 3'
    if seg_image.ndim > 2:
        if dataformats == 'CHW':
            c, h, w = seg_image.size()
            seg_image = seg_image.argmax(dim=0, keepdim=False) if c > 1 else seg_image
        elif dataformats == 'HWC':
            h, w, c = seg_image.size()
            seg_image = seg_image.argmax(dim=2, keepdim=False) if c > 1 else seg_image
        #
    #
    if isinstance(seg_image, torch.Tensor):
        seg_image = seg_image.cpu().numpy()
    #
    return xnn.utils.segmap_to_color(seg_image, num_classes)


def evaluate(args, model, data_loader, device, num_classes, epoch, visualizer):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    epoch_size = len(data_loader)
    visualization_freq = epoch_size // 5
    visualization_counter = 0
    print_freq = min(args.print_freq, len(data_loader))

    with torch.no_grad():
        for iteration, (image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

            if args.tensorboard and (visualizer is not None) and (iteration % visualization_freq) == 0:
                target_color = tensor_to_color(target[0], num_classes)
                output_color = tensor_to_color(output[0], num_classes)
                dataformats_color = 'HWC'
                write_visualization(visualizer, f'val-image/{visualization_counter}', image[0], epoch,
                    image_mean=np.array(args.image_mean), image_scale=np.array(args.image_scale), image_formatting=True)
                write_visualization(visualizer, f'val-target/{visualization_counter}', target_color, epoch, dataformats=dataformats_color)
                write_visualization(visualizer, f'val-output/{visualization_counter}', output_color, epoch, dataformats=dataformats_color)
                visualization_counter = visualization_counter + 1
            #
        #
        confmat.reduce_from_all_processes()

        metric_key, metric_value = list(confmat.metric().items())[0]
        if visualizer is not None:
            visualizer.add_scalar(f'val-metric/epoch-{metric_key}', metric_value, epoch)
        #
    #
    return confmat


def train_one_epoch(args, model, criterion, optimizer, data_loader, lr_scheduler, device, num_classes, epoch, visualizer):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)

    epoch_size = len(data_loader)
    visualization_freq = epoch_size // 5
    visualization_counter = 0
    print_freq = min(args.print_freq, len(data_loader))
    
    for iteration, (image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        if args.tensorboard and (iteration % visualization_freq) == 0 and (visualizer is not None):
            target_color = tensor_to_color(target[0], num_classes)
            output_color = tensor_to_color(output['out'][0], num_classes)
            dataformats_color = 'HWC'
            visualizer.add_scalar(f'train-loss/iter', loss.item(), epoch*epoch_size+iteration)
            write_visualization(visualizer, f'train-image/{visualization_counter}', image[0], epoch,
                image_mean=np.array(args.image_mean), image_scale=np.array(args.image_scale), image_formatting=True)
            write_visualization(visualizer, f'train-target/{visualization_counter}', target_color, epoch, dataformats=dataformats_color)
            write_visualization(visualizer, f'train-output/{visualization_counter}', output_color, epoch, dataformats=dataformats_color)
            visualization_counter = visualization_counter + 1
        #
    #
    if visualizer is not None:
        visualizer.add_scalar(f'train-loss/epoch', metric_logger.meters['loss'].avg, epoch)


def export(args, model, model_name=None):
    model.eval()
    model_name = 'model' if model_name is None else model_name
    image_size_tuple = args.input_size if isinstance(args.input_size, (list,tuple)) else \
        (args.input_size, args.input_size)
    data_shape = (1,3,*image_size_tuple)
    example_input = torch.rand(*data_shape)
    output_onnx_file = os.path.join(args.output_dir, 'model.onnx')
    torch.onnx.export(model, example_input, output_onnx_file, opset_version=args.opset_version)
    # shape inference to make it easy for inference
    onnx.shape_inference.infer_shapes_path(output_onnx_file, output_onnx_file)
    # export torchscript model
    # script_model = torch.jit.trace(model, example_input, strict=False)
    # torch.jit.save(script_model, os.path.join(args.output_dir, model_name+'_model.pth'))


def complexity(args, model):
    model.eval()
    image_size_tuple = args.input_size if isinstance(args.input_size, (list,tuple)) else \
        (args.input_size, args.input_size)
    data_shape = (1,3,*image_size_tuple)
    device = next(model.parameters()).device
    try:
        torchinfo.summary(model, data_shape, depth=10, device=device)
    except UnicodeEncodeError:
        warnings.warn('torchinfo.summary could not print - please check language/locale')


def main(gpu, args):
    if args.device != 'cpu' and args.distributed is True:
        os.environ['RANK'] = str(int(os.environ['RANK'])*args.gpus + gpu) if 'RANK' in os.environ else str(gpu)
        os.environ['LOCAL_RANK'] = str(gpu)
	
    if args.resize_with_scale_factor:
        nn.functional._interpolate_orig = nn.functional.interpolate
        nn.functional.interpolate = xnn.layers.resize_with_scale_factor

    if args.output_dir is None:
        args.output_dir = os.path.join('./data/checkpoints/segmentation', f'{args.dataset}_{args.model}')

    utils.mkdir(args.output_dir)
    logger = xnn.utils.TeeLogger(os.path.join(args.output_dir, f'run_{args.date}.log'))

    if args.device != 'cpu':
        utils.init_distributed_mode(args)
    else:
        args.distributed = False

    # print args
    print(f'{Fore.YELLOW}')
    print(args)

    # input pre-processing - show in pixel domain - for information
    print(f'preproc mean : {[round_frac(m) for m in args.image_mean]}')
    print(f'preproc scale: {[round_frac(s) for s in args.image_scale]}')
    print(f'{Fore.RESET}')

    train_visualizer = None
    val_visualizer = None

    if args.tensorboard and utils.is_main_process() and (not args.test_only) and (not args.export_only):
        from torch.utils.tensorboard import SummaryWriter
        train_vis_dir = os.path.join(args.output_dir, 'tensorboard', 'train')
        os.makedirs(train_vis_dir, exist_ok=True)
        train_visualizer = SummaryWriter(train_vis_dir, flush_secs=30)
        val_vis_dir = os.path.join(args.output_dir, 'tensorboard', 'val')
        os.makedirs(val_vis_dir, exist_ok=True)
        val_visualizer = SummaryWriter(val_vis_dir, flush_secs=30)

    device = torch.device(args.device)

    dataset, num_classes = get_dataset(args.data_path, args.dataset, "train", get_transform(args, train=True))
    dataset_test, _ = get_dataset(args.data_path, args.dataset, "val", get_transform(args, train=False))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    model = torchvision.models.segmentation.__dict__[args.model](num_classes=num_classes,
                                                                 aux_loss=args.aux_loss,
                                                                 pretrained=args.pretrained,
                                                                 pretrained_backbone=args.pretrained_backbone)

    if args.export_only:
        if args.distributed is False or (args.distributed is True and int(os.environ['LOCAL_RANK']) == 0):
            export(args, model, args.model)
            return

    if args.complexity:
        complexity(args, model)

    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    if args.aux_loss and hasattr(model_without_ddp, 'aux_classifier'):
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    iters_per_epoch = len(data_loader)
    main_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (iters_per_epoch * (args.epochs - args.lr_warmup_epochs))) ** 0.9)

    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == 'linear':
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.lr_warmup_decay,
                                                                    total_iters=warmup_iters)
        elif args.lr_warmup_method == 'constant':
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=args.lr_warmup_decay,
                                                                      total_iters=warmup_iters)
        else:
            raise RuntimeError("Invalid warmup lr method '{}'. Only linear and constant "
                               "are supported.".format(args.lr_warmup_method))
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        confmat = evaluate(args, model, data_loader_test, device=device, num_classes=num_classes, epoch=0, visualizer=None)
        print(confmat, '\n\n')
        return

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(args, model, criterion, optimizer, data_loader, lr_scheduler, device, num_classes=num_classes, epoch=epoch, visualizer=train_visualizer)
        print(f'{Fore.GREEN}', end='')
        confmat = evaluate(args, model, data_loader_test, device=device, num_classes=num_classes, epoch=epoch, visualizer=val_visualizer)
        print(confmat, '\n\n')
        print(f'{Fore.RESET}', end='')
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args
        }
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training', add_help=add_help)

    parser.add_argument('--data-path', default='./data/datasets/coco', help='dataset path')
    parser.add_argument('--dataset', default='coco', help='dataset name')
    parser.add_argument('--model', default='deeplabv3plus_mobilenet_v2_lite', help='model')
    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.05, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=4e-5, type=float,
                        metavar='W', help='weight decay (default: 4e-5)',
                        dest='weight_decay')
    parser.add_argument('--lr-warmup-epochs', default=0, type=int, help='the number of epochs to warmup (default: 0)')
    parser.add_argument('--lr-warmup-method', default="linear", type=str, help='the warmup method (default: linear)')
    parser.add_argument('--lr-warmup-decay', default=0.01, type=float, help='the decay for lr')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--output-dir', default=None, help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=None,
        type=xnn.utils.str_or_bool,
        help="Pre-trained models path or use from from the modelzoo",
    )
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--distributed", default=None, type=xnn.utils.str2bool_or_none,
                        help="use dstributed training even if this script is not launched using torch.disctibuted.launch or run")

    parser.add_argument('--gpus', default=1, type=int, help='number of gpus')
    parser.add_argument('--complexity', default=True, type=xnn.utils.str2bool, help='display complexity')
    parser.add_argument('--date', default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), help='current date')
    parser.add_argument('--input-size', default=(512,512), type=int, nargs='*', help='resized image size or the smaller side')
    parser.add_argument('--crop-size', default=(480,480), type=int, nargs='*', help='cropped image size for training')
    parser.add_argument('--opset-version', default=11, type=int, nargs='*', help='opset version for onnx export')
    parser.add_argument('--resize-with-scale-factor', type=xnn.utils.str2bool, default=True, help='resize with scale factor')
    parser.add_argument('--data-augmentation', default=None, type=xnn.utils.str_or_none, help='type of data augmentation')
    parser.add_argument('--image-mean', default=[123.675, 116.28, 103.53], type=float, nargs=3, help='mean subtraction of input')
    parser.add_argument('--image-scale', default=[0.017125, 0.017507, 0.017429], type=float, nargs=3, help='scale for multiplication of input')
    parser.add_argument(
        "--pretrained-backbone",
        dest="pretrained_backbone",
        default=True,
        type=xnn.utils.str_or_bool,
        help="Pre-trained backbone path or use from from the modelzoo",
    )
    parser.add_argument(
        "--export-only",
        dest="export_only",
        help="Only export the model",
        action="store_true",
    )
    parser.add_argument('--tensorboard', default=True, type=xnn.utils.str_or_bool, help='will use tensorboard if specified')
    return parser


def run(args):
    if isinstance(args.input_size, (list,tuple)) and len(args.input_size) == 1:
        args.input_size = args.input_size[0]

    if args.device != 'cpu' and args.distributed is True:
        # for explanation of what is happening here, please see this:
        # https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
        # this assignment of RANK assumes a single machine, but with multiple gpus
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = str(args.gpus)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        torch.multiprocessing.spawn(main, nprocs=args.gpus, args=(args,))
    else:
        main(0, args)


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    if isinstance(args.input_size, (list,tuple)) and len(args.input_size) == 1:
        args.input_size = args.input_size[0]

    if isinstance(args.crop_size, (list,tuple)) and len(args.crop_size) == 1:
        args.crop_size = args.crop_size[0]

    # run the training.
    # if args.distributed is True is set, then this will launch distributed training
    # depending on args.gpus
    run(args)
	
