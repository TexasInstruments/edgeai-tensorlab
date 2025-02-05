
_base_ = '../../configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_tiny_4xb64-300e_coco-416.py'

convert_to_lite_model = dict(model_surgery=1)
find_unused_parameters=True

load_from = '../edgeai-modelzoo/models/vision/keypoint/coco/edgeai-mmpose/yoloxpose_tiny_lite_416x416_mmpose_original_checkpoint.pth'

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer = dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2)
    )
