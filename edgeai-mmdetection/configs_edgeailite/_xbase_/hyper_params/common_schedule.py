# optimizer
optimizer = dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=1e-4)

# gradient clipping options
#optimizer_config = dict(grad_clip=None)
#optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# warmup by iterations
# warmup_cfg = dict(warmup='linear', warmup_iters=1000, warmup_ratio=1e-3)
# warmup by epoch
warmup_cfg = dict(warmup='linear', warmup_by_epoch=True, warmup_iters=1, warmup_ratio=1e-4)

# lr policy - step
# lr_config = dict(policy='step', step=[40, 55], **warmup_cfg)
# lr policy - cosine
lr_config = dict(policy='CosineAnnealing', min_lr_ratio=1e-4, **warmup_cfg)

# runtime settings
total_epochs = 240 #120 #60

# enable fp16 - optional
#fp16 = dict(loss_scale=512.0)

