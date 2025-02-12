_base_ = ['../../configs/yolov9/yolov9_tiny_coco.py']

model = dict(
    backbone=dict(
        pool_kernel_size=3,
        ),
    neck=dict(
        pool_kernel_size=3,
        ))