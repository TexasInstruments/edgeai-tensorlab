_base_ = ['../../configs/yolov7/yolov7_l_coco.py']
# model settings

convert_to_lite_model = dict(model_surgery=1)

#load_from = '../edgeai-modelzoo/models/vision/detection/coco/edgeai-mmdet/yolov7_l_coco_origin_640x640_20250109_checkpoint.pth'

