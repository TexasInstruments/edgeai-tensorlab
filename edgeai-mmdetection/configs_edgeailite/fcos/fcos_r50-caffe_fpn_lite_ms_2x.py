_base_ = '../../configs/fcos/fcos_r50-caffe_fpn_gn-head_ms-640-800-2x_coco.py'

# replace complex activation functions with ReLU.
# Also, if needed, regular convolutions can be replaced with depthwise-separable convolutions.
# edgeai_torchmodelopt needs to be installed from edgeai-modeloptimization
#convert_to_lite_model = dict(model_surgery=1)

# use scale factors (instead of output size) in Resize layers
resize_with_scale_factor = True

# for some reason, not able to train after doing the above model surgery - so manually replace
# use BN instead of GN/InstanceNorm
model = dict(
    bbox_head=dict(
        norm_cfg=dict(type='BN')
    )
)

