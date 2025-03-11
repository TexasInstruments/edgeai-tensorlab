_base_ = './fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_tidl.py'

train_cfg = dict(max_epochs=10)
optim_wrapper = dict(
    # type='AmpOptimWrapper', loss_scale=32.,#'dynamic', # for fp16
    optimizer=dict(lr=1e-5),
)
model = dict(
    save_onnx_model=False,
    train_cfg=dict(
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05]
        )
    )

env_cfg = dict(
    dist_cfg = dict(timeout=3600)
)

train_dataloader = dict( batch_size=2, num_workers=4)
test_dataloader = dict( batch_size=1, num_workers=4)
val_dataloader = dict( batch_size=1, num_workers=4)

load_from = './checkpoints/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth'

find_unused_parameters = True

# fp16 settings
# fp16 = dict(loss_scale=32.)#'dynamic')