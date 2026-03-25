_base_ = '../../configs/centernet/centernet-update_r50-caffe_fpn_ms-1x_coco.py'

# use scale factors (instead of output size) in Resize layers
resize_with_scale_factor = True

# for some reason, not able to train after doing the above model surgery - so manually replace
# use BN instead of GN/InstanceNorm
model = dict(
    bbox_head=dict(
        norm_cfg=dict(type='BN')
    )
)
