
_base_ = '../../configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_tiny_4xb64-300e_coco-416.py'

convert_to_lite_model = dict(model_surgery=1)
find_unused_parameters=True

load_from = '/data/files/a0508577/work/edgeai-algo/edgeai-mmpose/work_dirs/' \
    'checkpoints/yoloxpose_tiny_4xb64-300e_coco-416-76eb44ca_20230829_new.pth'
