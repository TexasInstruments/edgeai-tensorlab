_base_ = '../configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_s_8xb32-300e_coco-640.py'

load_from = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_s_8xb32-300e_coco-640-56c79c1f_20230829.pth'

convert_to_lite_model = dict(model_surgery=1)
find_unused_parameters=True

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2)
    )


