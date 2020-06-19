
checkpoint_config = dict(interval=1)

# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# optimizer
optimizer = dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=1e-4)

# gradient clipping - adopted from fcos config
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(policy='CosineAnealing', min_lr_ratio=1e-3, warmup='linear', warmup_iters=1000, warmup_ratio=1e-4)

# runtime settings
# recommend: 12, 24 or 48
total_epochs = 24

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
print_model_complexity = True
save_model_proto = True

