cudnn_benchmark = True
default_scope = 'mmdet'

# replace torch.nn.functional.interpolate with edgeai_torchmodelopt.xnn.layers.resize_with_scale_factor
# edgeai_torchmodelopt needs to be installed from edgeai-modeloptimization
resize_with_scale_factor = True

# replace complex activation functions with ReLU.
# Also replace regular convolutions with depthwise-separable convolutions.
# edgeai_torchmodelopt needs to be installed from edgeai-modeloptimization
# Note: this has now n
# convert_to_lite_model = dict(group_size_dw=1)

checkpoint_config = dict(interval=1)

# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
print_model_complexity = True

# fp16 settings
#fp16 = dict(loss_scale=512.)
#fp16 = dict(loss_scale='dynamic')
