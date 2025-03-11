_base_ = ['../../projects/EfficientDet/configs/efficientdet_effb1_bifpn_8xb16-crop512-300e_coco.py']

convert_to_lite_model = dict(model_surgery=1)

model = dict(
    backbone=dict(
        conv_cfg=dict(type='Conv2dSamePaddingDefault')),
    neck=dict(
        conv_cfg=dict(type='DepthWiseConvBlockDefaultPadding')),
    bbox_head=dict(
        conv_cfg=dict(type='DepthWiseConvBlockDefaultPadding')),
    test_cfg=dict(
        nms=dict(
            _delete_=True,
            type='nms',
            iou_threshold=0.3
            )))
    