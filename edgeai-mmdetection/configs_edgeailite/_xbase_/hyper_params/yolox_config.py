
cudnn_benchmark = True
default_scope = 'mmdet'


# replace torch.nn.functional.interpolate with edgeai_torchmodelopt.xnn.layers.resize_with_scale_factor
# edgeai_torchmodelopt needs to be installed from edgeai-modeloptimization
resize_with_scale_factor = True

# yapf:enable
max_epochs = 300
num_last_epochs = 15
interval = 10
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
print_model_complexity = True

# fp16 settings
#fp16 = dict(loss_scale=512.)
#fp16 = dict(loss_scale='dynamic')



# optimizer
# default 8 gpu
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

# runner = dict(
#     type='EpochBasedRunner', max_epochs=max_epochs)



custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        # num_last_epochs=num_last_epochs,
        # interval=interval,
        priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]

checkpoint_config = dict(
    interval=interval)

log_config = dict(
    interval=100, hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
