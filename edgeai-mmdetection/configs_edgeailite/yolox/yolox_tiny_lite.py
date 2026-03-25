_base_ = '../../configs/yolox/yolox_tiny_8xb8-300e_coco.py'

# replace complex activation functions with ReLU.
# Also, if needed, regular convolutions can be replaced with depthwise-separable convolutions.
# edgeai_torchmodelopt needs to be installed from edgeai-modeloptimization
convert_to_lite_model = dict(model_surgery=1)

model = dict(
    backbone=dict(act_cfg=dict(type='ReLU')),
    neck=dict(act_cfg=dict(type='ReLU')),
    bbox_head=dict(act_cfg=dict(type='ReLU')))
