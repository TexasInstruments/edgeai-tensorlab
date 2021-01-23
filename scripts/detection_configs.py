'''
Usage:
(1) Use one of the following config files.
(2) Inside the config file, make sure that the dataset that needs to be trained on is uncommented.
(3) Use the appropriate input resolution in the config file (input_size).
(4) Recommend to run the first training with voc0712 dataset as it is widely used and reasonably small.
(5) To convert cityscapes to coco format, run the script: tools/convert_datasets/cityscapes.py

config='./configs/ssd/ssd_resnet_fpn.py'
config='./configs/ssd/ssd-lite_mobilenet.py'
config='./configs/ssd/ssd-lite_mobilenet_fpn.py'
config='./configs/ssd/ssd-lite_regnet_fpn_bgr.py'

config='./configs/retinanet/retinanet_resnet_fpn.py'
config='./configs/retinanet/retinanet-lite_regnet_fpn_bgr.py'

config='./configs/yolo/yolov3_d53.py'
config='./configs/yolo/yolov3_d53_relu.py'
config='./configs/yolo/yolov3-lite_d53.py'
'''

config='./configs/ssd/ssd-lite_regnet_fpn_bgr.py'

