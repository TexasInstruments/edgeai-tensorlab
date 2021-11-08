import datetime
import os
import time
from colorama import Fore
import numpy as np
import onnx

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision.edgeailite import xnn
import torchinfo

from coco_utils import get_coco
import presets
import utils


def get_dataset(dir_path, name, image_set, transform):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)
    paths = {
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco": (dir_path, get_coco, 21)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(args, train):
    base_size = args.input_size
    crop_size = args.crop_size
    return presets.SegmentationPresetTrain(base_size, crop_size, mean=args.mean, scale=args.scale, data_augmentation=args.data_augmentation) \
        if train else presets.SegmentationPresetEval(base_size, mean=args.mean, scale=args.scale)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def round_frac(number, digits=4):
    return round(number, digits)


def tensor_to_numpy(tensor, mean=None, scale=None, dataformats=None):
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
    array = array / scale.reshape(view_args) if scale is not None else array
    array = array + mean.reshape(view_args) if scale is not None else array
    return array


def write_visualization(visualizer, name, tensor, epoch, mean=None, scale=None, image_formatting=None, dataformats=None):
    if dataformats is None:
        dataformats = 'HW' if tensor.ndim == 2 else ('CHW' if tensor.ndim == 3 else dataformats)
    #
    tensor = tensor_to_numpy(tensor, mean=mean, scale=scale, dataformats=dataformats)
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

    with torch.no_grad():
        for iteration, (image, target) in enumerate(metric_logger.log_every(data_loader, 100, header)):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

            if args.tensorboard and (visualizer is not None) and (iteration % visualization_freq) == 0:
                target_color = tensor_to_color(target[0], num_classes)
                output_color = tensor_to_color(output[0], num_classes)
                dataformats_color = 'HWC'
                write_visualization(visualizer, f'val-image/{visualization_counter}', image[0], epoch,
                    mean=np.array(args.mean), scale=np.array(args.scale), image_formatting=True)
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

    for iteration, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        if args.tensorboard and (iteration % visualization_freq) == 0:
            target_color = tensor_to_color(target[0], num_classes)
            output_color = tensor_to_color(output['out'][0], num_classes)
            dataformats_color = 'HWC'
            visualizer.add_scalar(f'train-loss/iter', loss.item(), epoch*epoch_size+iteration)
            write_visualization(visualizer, f'train-image/{visualization_counter}', image[0], epoch,
                mean=np.array(args.mean), scale=np.array(args.scale), image_formatting=True)
            write_visualization(visualizer, f'train-target/{visualization_counter}', target_color, epoch, dataformats=dataformats_color)
            write_visualization(visualizer, f'train-output/{visualization_counter}', output_color, epoch, dataformats=dataformats_color)
            visualization_counter = visualization_counter + 1
        #
    #
    visualizer.add_scalar(f'train-loss/epoch', metric_logger.meters['loss'].avg, epoch)


def export(args, model, model_name=None):
    model.eval()
    model_name = 'model' if model_name is None else model_name
    image_size_tuple = args.input_size if isinstance(args.input_size, (list,tuple)) else \
        (args.input_size, args.input_size)
    data_shape = (1,3,*image_size_tuple)
    example_input = torch.rand(*data_shape)
    output_onnx_file = os.path.join(args.output_dir, model_name+'.onnx')
    torch.onnx.export(model, example_input, output_onnx_file, opset_version=args.opset_version)
    onnx.shape_inference.infer_shapes_path(output_onnx_file, output_onnx_file)

    #script_model = torch.jit.trace(model, example_input, strict=False)
    #torch.jit.save(script_model, os.path.join(args.output_dir, model_name+'_model.pth'))


def complexity(args, model):
    model.eval()
    image_size_tuple = args.input_size if isinstance(args.input_size, (list,tuple)) else \
        (args.input_size, args.input_size)
    data_shape = (1,3,*image_size_tuple)
    torchinfo.summary(model, data_shape, depth=10)


def main(args):
    if args.resize_with_scale_factor:
        nn.functional._interpolate_orig = nn.functional.interpolate
        nn.functional.interpolate = xnn.layers.resize_with_scale_factor

    if args.output_dir:
        utils.mkdir(args.output_dir)
        logger = xnn.utils.TeeLogger(os.path.join(args.output_dir, f'run_{args.date}.log'))

    utils.init_distributed_mode(args)

    # print args
    print(f'{Fore.YELLOW}')
    print(args)

    # input pre-processing - show in pixel domain - for information
    print(f'preproc mean : {[round_frac(m) for m in args.mean]}')
    print(f'preproc scale: {[round_frac(s) for s in args.scale]}')
    print(f'{Fore.RESET}')

    if args.tensorboard and (not args.test_only) and (not args.export_only):
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

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

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
        train_one_epoch(args, model, criterion, optimizer, data_loader, lr_scheduler, device, num_classes=num_classes,
                        epoch=epoch, visualizer=train_visualizer)
        print(f'{Fore.GREEN}', end='')
        confmat = evaluate(args, model, data_loader_test, device=device, num_classes=num_classes,
                           epoch=epoch, visualizer=val_visualizer)
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

    parser.add_argument('--data-path', default=None, help='dataset path')
    parser.add_argument('--dataset', default=None, help='dataset name')
    parser.add_argument('--model', default=None, help='model')
    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./data/checkpoints/segmentation', help='path where to save')
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

    parser.add_argument('--gpus', default=1, type=int, help='number of gpus')
    parser.add_argument('--complexity', default=True, type=xnn.utils.str2bool, help='display complexity')
    parser.add_argument('--date', default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), help='current date')
    parser.add_argument('--input-size', default=520, type=int, nargs='*', help='resized image size or the smaller side')
    parser.add_argument('--crop-size', default=480, type=int, nargs='*', help='cropped image size for training')
    parser.add_argument('--opset-version', default=11, type=int, nargs='*', help='opset version for onnx export')
    parser.add_argument('--resize-with-scale-factor', action='store_true', help='resize with scale factor')
    parser.add_argument('--data-augmentation', default=None, type=xnn.utils.str_or_none, help='type of data augmentation')
    parser.add_argument('--mean', default=[123.675, 116.28, 103.53], type=float, nargs=3, help='mean subtraction')
    parser.add_argument('--scale', default=[0.017125, 0.017507, 0.017429], type=float, nargs=3, help='standard deviation for division')
    parser.add_argument(
        "--pretrained-backbone",
        dest="pretrained_backbone",
        default=None,
        type=xnn.utils.str_or_bool,
        help="Pre-trained backbone path or use from from the modelzoo",
    )
    parser.add_argument(
        "--export-only",
        dest="export_only",
        help="Only export the model",
        action="store_true",
    )
    parser.add_argument('--tensorboard', action='store_true', help='will use tensorboard if specified')
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    if isinstance(args.input_size, (list,tuple)) and len(args.input_size) == 1:
        args.input_size = args.input_size[0]

    if isinstance(args.crop_size, (list,tuple)) and len(args.crop_size) == 1:
        args.crop_size = args.crop_size[0]

    main(args)
