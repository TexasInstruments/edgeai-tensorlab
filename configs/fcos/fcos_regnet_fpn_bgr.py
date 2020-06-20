_base_ = [
    '../_jacinto_ai_base_/hyper_params/common_config.py',
    '../_jacinto_ai_base_/hyper_params/ssd_config.py',
]

dataset_type = 'VOCDataset'

if dataset_type == 'VOCDataset':
    _base_ += ['../_jacinto_ai_base_/datasets/voc0712_det.py']
    num_classes = 20
elif dataset_type == 'CocoDataset':
    _base_ += ['../_jacinto_ai_base_/datasets/coco_det.py']
    num_classes = 80
elif dataset_type == 'CityscapesDataset':
    _base_ += ['../_jacinto_ai_base_/datasets/cityscapes_det.py']
    num_classes = 8
else:
    assert False, f'Unknown dataset_type: {dataset_type}'


input_size = (768,384)                           # (1536,768) #(1024,512) #(768,384) #(512,512)
decoder_width_fact = 1                           # 1, 2, 3

backbone_type = 'RegNet'
backbone_arch = 'regnetx_800mf'                  # 'regnetx_800mf' #'regnetx_1.6gf'
to_rgb = False                                   # pycls regnet backbones are trained with bgr

regnet_settings = {
    'regnetx_800mf':{'regnet_base_channels':32, 'bacbone_out_channels':[64, 128, 288, 672],
                      'group_size_dw':16, 'fpn_out_channels':(64*decoder_width_fact),
                      'pretrained':'open-mmlab://regnetx_800mf'},
    'regnetx_1.6gf':{'regnet_base_channels':32, 'bacbone_out_channels':[72, 168, 408, 912],
                     'group_size_dw':24, 'fpn_out_channels':(96*decoder_width_fact),
                     'pretrained':'open-mmlab://regnetx_1.6gf'}}

regnet_cfg = regnet_settings[backbone_arch]
pretrained=regnet_cfg['pretrained']
regnet_base_channels=regnet_cfg['regnet_base_channels']
bacbone_out_channels=regnet_cfg['bacbone_out_channels']
backbone_out_indices = (0, 1, 2, 3)

fpn_in_channels = bacbone_out_channels
fpn_out_channels = regnet_cfg['fpn_out_channels']
fpn_start_level = 1
fpn_num_outs = 5

fcos_num_levels = 5
fcos_base_stride = (8 if fpn_start_level==1 else (4 if fpn_start_level==0 else None))

# for multi-scale training
input_size_ms = [(input_size[0], (input_size[1]*8)//10),
                 (input_size[0], (input_size[1]*9)//10),
                 input_size]

conv_cfg = dict(type='ConvDWSep', group_size_dw=regnet_cfg['group_size_dw'])
norm_cfg = dict(type='BN')
img_norm_cfg = dict(mean=[128.0, 128.0, 128.0], std=[64.0, 64.0, 64.0], to_rgb=to_rgb)

if fcos_num_levels > 1:
    fcos_input_size_max_edge = max(input_size)//2
    fcos_regress_range_start = fcos_input_size_max_edge//(2**(fcos_num_levels-1))
    fcos_pow2_factors = [2**i for i in range(fcos_num_levels)]
    fcos_regress_range_edges = [-1] + [fcos_regress_range_start*p2 for p2 in fcos_pow2_factors[1:]] + [1e8]
    fcos_regress_ranges = tuple([(fcos_regress_range_edges[i],fcos_regress_range_edges[i+1]) for i in range(fcos_num_levels)])
    fpn_num_outs = max(fcos_num_levels, len(backbone_out_indices))
    fcos_head_strides = [fcos_base_stride*p2 for p2 in fcos_pow2_factors]
else:
    fcos_regress_ranges = ((-1,1e8),)
    fpn_num_outs = len(backbone_out_indices)
    fcos_head_strides = [fcos_base_stride]
#

model = dict(
    type='FCOS',
    pretrained=pretrained,
    backbone=dict(
        type=backbone_type,
        arch=backbone_arch,
        base_channels=regnet_base_channels,
        out_indices=backbone_out_indices,
        norm_eval=False,
        style='pytorch'),
    neck=dict(
        type='JaiFPN',
        in_channels=fpn_in_channels,
        out_channels=fpn_out_channels,
        start_level=fpn_start_level,
        num_outs=fpn_num_outs,
        add_extra_convs='on_input', #'on_output',
        upsample_cfg=dict(scale_factor=2, mode='bilinear'), #,mode='nearest'),
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='JaiFCOSHead',
        num_classes=num_classes,
        in_channels=fpn_out_channels,
        stacked_convs=4,
        feat_channels=fpn_out_channels,
        strides=fcos_head_strides,
        regress_ranges=fcos_regress_ranges,
        center_sample_radius=1.5,
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=input_size_ms,
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=input_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=3,
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# settings for qat or calibration - uncomment after doing floating point training
# also change dataset_repeats in the dataset config to 1 for fast learning
quantize = False #'training' #'calibration'
if quantize:
  load_from = './work_dirs/fcos_regnet_fpn/latest.pth'
  optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=1e-4)
  lr_config = dict(policy='CosineAnealing', min_lr_ratio=1e-3, warmup='linear', warmup_iters=100, warmup_ratio=1e-4)
  total_epochs = 1 if quantize == 'calibration' else 5
#


