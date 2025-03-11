_base_ = ['./yolov3_d53.py']

act_cfg = dict(type='ReLU')
# model settings
model = dict(
    backbone=dict(
        act_cfg=act_cfg
    ),
    neck=dict(
        act_cfg=act_cfg),
    bbox_head=dict(
        act_cfg=act_cfg
    ))