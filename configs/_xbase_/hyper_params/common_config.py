cudnn_benchmark = True

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
