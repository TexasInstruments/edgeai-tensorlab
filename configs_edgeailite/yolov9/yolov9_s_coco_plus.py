_base_ = ['../../configs/yolov9/yolov9_s_coco.py']

load_from = '../edgeai-modelzoo/models/vision/detection/coco/edgeai-mmdet/yolov9_s_coco_origin_640x640_20250113_checkpoint.pth'

model = dict(
    backbone=dict(
        pool_kernel_size=3,
        ),
    neck=dict(
        pool_kernel_size=3,
        ))