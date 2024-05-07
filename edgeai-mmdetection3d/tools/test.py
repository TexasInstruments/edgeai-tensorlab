# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils.runner import XMMDetEpochBasedRunner

if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg
from mmdet3d.utils.proto import tidl_meta_arch_pb2
from google.protobuf import text_format

class CombinedModel(torch.nn.Module):

    def __init__(self, pfn_layers, middle_encoder, backbone, neck, conv_cls, conv_dir_cls,conv_reg,num_class):
        super().__init__()
        self.pfn_layers     = pfn_layers
        self.middle_encoder = middle_encoder
        self.backbone       = backbone
        self.neck           = neck
        self.conv_cls       = conv_cls
        self.conv_dir_cls   = conv_dir_cls
        self.conv_reg       = conv_reg
        self.num_ancohors   = num_class * 2  # for each classs two anchors at 90 degree of placement is used

    def forward(self, raw_voxel_feat, coors, data):

        x = self.pfn_layers(raw_voxel_feat,ip_tensor_dim_correct=True)
        x = self.middle_encoder(x,coors,1,data)
        x = self.backbone(x)
        x = self.neck(x)
        y0= self.conv_cls(x[0])
        y1= self.conv_dir_cls(x[0])
        y2= self.conv_reg(x[0])
        y = torch.cat((y0, y1, y2), 1)
        y = torch.reshape(y,(-1,(int)((y0.shape[-3]+y1.shape[-3]+y2.shape[-3])/self.num_ancohors)))
        return y

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    if hasattr(cfg,'save_onnx_model') is False:
        cfg.save_onnx_model = False

    if hasattr(cfg,'quantize') is False:
        cfg.quantize = False

    if hasattr(cfg,'match_tidl_nms') is False:
        cfg.match_tidl_nms = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    if cfg.quantize == True:
        from mmdet3d.utils.quantize import XMMDetQuantTrainModule
        import numpy as np

        raw_voxel_feat = np.fromfile("./data/kitti/training/velodyne/000000.bin")
        raw_voxel_feat = torch.tensor(raw_voxel_feat.reshape((-1,4)))

        model = XMMDetQuantTrainModule(model, [raw_voxel_feat,raw_voxel_feat])

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    if cfg.quantize == True:        
        checkpoint = load_checkpoint(model.module, args.checkpoint, map_location='cpu')
    else:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    import torch.nn.utils.prune as prune

    def parseConv(module, prefix=''):

        if isinstance(module, torch.nn.Conv2d):
            #prune.l1_unstructured(module, name='weight', amount=0.1)
            prune.ln_structured(module, name="weight", amount=0.1, n=float('-inf'), dim=1)
            print("Sparsity in {} {:.2f}%".format(prefix+"Conv2d",
                    100. * float(torch.sum(module.weight == 0))
                    / float(module.weight.nelement())
                )
            )
            #prune.remove(module, 'weight')

        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.1)
            #prune.ln_structured(module, name="weight", amount=0.1, n=2, dim=3)
            print("Sparsity in {} {:.2f}%".format(prefix+"Linear",
                    100. * float(torch.sum(module.weight == 0))
                    / float(module.weight.nelement())
                )
            )
            #prune.remove(module, 'weight')

        for name, child in module._modules.items():
            if child is not None:
                parseConv(child, prefix + name + '.')

    sparsify_model = False
    if sparsify_model == True:
        parseConv(model)

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
         model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if cfg.save_onnx_model == True:
        save_onnx_model(cfg, model, os.path.dirname(args.checkpoint),cfg.quantize)
    
    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))

    if cfg.match_tidl_nms == True:
        print('TIDL doesnt support 3D NMS right now, hence to match TIDL NMS another evaluation is happeing now with upright rectangle NMS with threshold nms_th=0.5')
        cfg.model.test_cfg.use_rotate_nms = False
        cfg.model.test_cfg.nms_thr = 0.5
        if not distributed:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                    args.gpu_collect)

        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                mmcv.dump(outputs, args.out)
            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset.format_results(outputs, **kwargs)
            if args.eval:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in [
                        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                        'rule'
                ]:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))
                print(dataset.evaluate(outputs, **eval_kwargs))

def save_onnx_model(cfg, model,path,quantized_model=True,tag=''):

        print('onnx export started, at path {}'.format(path))

        model_cpu = model.to('cpu')

        #device_ids=[torch.cuda.current_device()]
        
        #if len(device_ids) > 1:
        #    print('Multiple GPUs visible, Currently only one GPU can be used for onnx export')
            
        from test import CombinedModel

        model_cpu.eval()

        bev_size_x = (int)((cfg.point_cloud_range[3] - cfg.point_cloud_range[0])/cfg.voxel_size[0])
        bev_size_y = (int)((cfg.point_cloud_range[4] - cfg.point_cloud_range[1])/cfg.voxel_size[1])

        # Accessing one layer to find wheather wegihts are on cpu or cuda
        #if quantized_model == True:
        #    device = model_cpu.module.voxel_encoder.pfn_layers._modules['0'].linear.weight.device
        #else:
        #    device = model_cpu.voxel_encoder.pfn_layers._modules['0'].linear.weight.device

        device ='cpu'

        raw_voxel_feat = torch.ones([1, cfg.model.voxel_encoder.in_channels+6, 
                                        cfg.model.voxel_layer.max_num_points, 
                                        cfg.model.voxel_layer.max_voxels[2]],
                                        device=torch.device(device))

        data = torch.zeros([1, cfg.model.voxel_encoder.feat_channels[0], bev_size_x*bev_size_y],device=torch.device(device))
        coors = torch.ones([1, cfg.model.voxel_encoder.feat_channels[0], cfg.model.voxel_layer.max_voxels[2]],device=torch.device(device))
        coors = coors.long()

        if quantized_model == True:
            combined_model = CombinedModel(model_cpu.module.voxel_encoder.pfn_layers._modules['0'],
                                            model_cpu.module.middle_encoder,
                                            model_cpu.module.backbone,
                                            model_cpu.module.neck,
                                            model_cpu.module.bbox_head.conv_cls,
                                            model_cpu.module.bbox_head.conv_dir_cls,
                                            model_cpu.module.bbox_head.conv_reg,
                                            len(model_cpu.CLASSES)
                                            )
        else:
            combined_model = CombinedModel(model_cpu.voxel_encoder.pfn_layers._modules['0'],
                                            model_cpu.middle_encoder,
                                            model_cpu.backbone,
                                            model_cpu.neck,
                                            model_cpu.bbox_head.conv_cls,
                                            model_cpu.bbox_head.conv_dir_cls,
                                            model_cpu.bbox_head.conv_reg,
                                            len(model_cpu.CLASSES)
                                            )

        torch.onnx.export(combined_model,
                (raw_voxel_feat,coors,data),
                os.path.join(path,tag+"combined_model.onnx"),
                opset_version=11,
                verbose=False)
                
        # change the data type from int64 to int32 for co-ordinates in scatter layer, as TIDL doesnt supprt that
        # it can not be changes in pytorch operator as int32 is not supported there.
        import onnx
        onnx_model = onnx.load(os.path.join(path,tag+"combined_model.onnx"))
        graph = onnx_model.graph

        for inp in graph.input:
            if inp.name == 'coors' and inp.type.tensor_type.elem_type == onnx.TensorProto.INT64 :
                inp.type.tensor_type.elem_type = onnx.TensorProto.INT32

        onnx.save(onnx_model, os.path.join(path,tag+"combined_model_int32.onnx"))
        save_tidl_prototxt(cfg, path, tag)

def save_tidl_prototxt(cfg, path,tag=''):

    import onnx
    prototxt_file = os.path.join(path,tag+"pointPillars.prototxt")
    model = onnx.load(os.path.join(path,tag+"combined_model_int32.onnx"))
    graph = model.graph
    
    box_input = []
    dir_input = []
    cls_input = []

    for node in graph.node:
        for inp in node.input:
            if 'conv_cls' in inp and (len(cls_input) == 0):
                cls_input.append(node.output)
            if 'conv_dir' in inp and (len(dir_input) == 0):
                dir_input.append(node.output)
            if 'conv_reg' in inp and (len(box_input) == 0):
                box_input.append(node.output)

    if cfg.quantize == True:
        for node in graph.node:
            if node.op_type == 'Clip':
                for inp in node.input:
                    if cls_input[0][0] == inp and (len(cls_input) <= 1):
                        cls_input = []
                        cls_input.append(node.output)
                    if dir_input[0][0] == inp and (len(dir_input) <= 1):
                        dir_input = []
                        dir_input.append(node.output)
                    if box_input[0][0] == inp and (len(box_input) <= 1):
                        box_input = []
                        box_input.append(node.output)


    bev_size_x = (int)((cfg.point_cloud_range[3] - cfg.point_cloud_range[0])/cfg.voxel_size[0])
    bev_size_y = (int)((cfg.point_cloud_range[4] - cfg.point_cloud_range[1])/cfg.voxel_size[1])

    offset_dir= -cfg.model.bbox_head.anchor_generator.rotations[1]    
    prior_box_param = []
    
    for id in range(cfg.model.bbox_head.num_classes):
        step_x = (cfg.model.bbox_head.anchor_generator.ranges[id][3] - cfg.model.bbox_head.anchor_generator.ranges[id][0])/(bev_size_x/2) 
        step_y = (cfg.model.bbox_head.anchor_generator.ranges[id][4] - cfg.model.bbox_head.anchor_generator.ranges[id][1])/(bev_size_y/2)   
        step_z = 1

        offset_x = (cfg.model.bbox_head.anchor_generator.ranges[id][0])/step_x + 0.5 # 0.5 = step_x/2 * step_x
        offset_y = (cfg.model.bbox_head.anchor_generator.ranges[id][1])/step_y + 0.5 # 0.5 = step_y/2 * step_y
        offset_z = (cfg.model.bbox_head.anchor_generator.ranges[id][2])/step_z + 0.0 # 0.0 = step_z/2 * step_z

        prior_box_param.append(tidl_meta_arch_pb2.PriorBox3DODParameter(anchor_width=[cfg.model.bbox_head.anchor_generator.sizes[id][0]], 
                                                                        anchor_height=[cfg.model.bbox_head.anchor_generator.sizes[id][1]], 
                                                                        anchor_length=[cfg.model.bbox_head.anchor_generator.sizes[id][2]],
                                                                        step_x=step_x, step_y=step_y, step_z=step_z,
                                                                        offset_x=offset_x, offset_y=offset_y, offset_z=offset_z,
                                                                        offset_dir = offset_dir,
                                                                        rotation=cfg.model.bbox_head.anchor_generator.rotations))
    
    nms_param = tidl_meta_arch_pb2.TIDLNmsParam(nms_threshold=cfg.model.test_cfg.nms_thr,top_k=cfg.model.test_cfg.nms_pre)
    
    detection_param   = tidl_meta_arch_pb2.TIDLOdPostProc(num_classes=cfg.model.bbox_head.num_classes,
                                                          share_location=True,
                                                          background_label_id=-1,
                                                          code_type="CODE_TYPE_3DOD",
                                                          keep_top_k=cfg.model.test_cfg.max_num,
                                                          confidence_threshold=cfg.model.test_cfg.score_thr,
                                                          nms_param=nms_param
                                                        )                                            

    
    od3d = tidl_meta_arch_pb2.TidlMa3DOD(name="point_pillars", 
                                            min_x=cfg.point_cloud_range[0],
                                            max_x=cfg.point_cloud_range[3],
                                            min_y=cfg.point_cloud_range[1],
                                            max_y=cfg.point_cloud_range[4],
                                            min_z=cfg.point_cloud_range[2], 
                                            max_z=cfg.point_cloud_range[5],
                                            voxel_size_x=cfg.voxel_size[0],voxel_size_y=cfg.voxel_size[1],voxel_size_z=cfg.voxel_size[2],
                                            max_points_per_voxel=cfg.model.voxel_layer.max_num_points,
                                            box_input=box_input[0],class_input=cls_input[0],dir_input=dir_input[0],
                                            output=[node.name for node in model.graph.output],
                                            prior_box_3dod_param=prior_box_param,
                                            detection_output_param=detection_param)

    arch = tidl_meta_arch_pb2.TIDLMetaArch(name='3dod_ssd',  tidl_3dod=[od3d])

    with open(prototxt_file, 'wt') as pfile:
        txt_message = text_format.MessageToString(arch)
        pfile.write(txt_message)

if __name__ == '__main__':
    main()
