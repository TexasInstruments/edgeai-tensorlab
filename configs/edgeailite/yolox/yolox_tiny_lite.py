
# modified from: https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox

_base_ = '../../yolox/yolox_tiny_8x8_300e_coco.py'

img_scale = (416, 416)
input_size = img_scale

# settings for qat or calibration - set to True after doing floating point training
quantize = False #'training' #'calibration'

img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=True)


# in the above base config, the image_scale for train_pipeline and test_pipeline are different.
# not sure how much it matters, but re-defining them here with matching values.

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.5, 1.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

if quantize:
    load_from = './work_dirs/yolox_s_lite/latest.pth'
    total_epochs = (1 if quantize == 'calibration' else 12)
    num_last_epochs = 1
else:
    load_from = None
    total_epochs = 240
    num_last_epochs = 15
#

runner = dict(max_epochs=total_epochs)

