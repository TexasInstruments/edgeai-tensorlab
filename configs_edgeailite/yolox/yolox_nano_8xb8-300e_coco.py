_base_ = './yolox_tiny_8xb8-300e_coco.py'

# replace complex activation functions with ReLU.
# Also, if needed, regular convolutions can be replaced with depthwise-separable convolutions.
# edgeai_torchmodelopt needs to be installed from edgeai-modeloptimization
# convert_to_lite_model = dict(model_surgery=1)

# model settings
model = dict(
    backbone=dict(deepen_factor=0.33, widen_factor=0.25, use_depthwise=True),
    neck=dict(
        in_channels=[64, 128, 256],
        out_channels=64,
        num_csp_blocks=1,
        use_depthwise=True),
    bbox_head=dict(in_channels=64, feat_channels=64, use_depthwise=True))
