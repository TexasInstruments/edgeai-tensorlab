#!/usr/bin/env python

import sys
import os
from torch.distributed import launch as distributed_launch

########################################################################
# config

'''
Usage: 
(1) Use one of the following config files.
(2) Inside the config file, make sure that the dataset that needs to be trained on is uncommented.
(3) Use the appropriate input resolution int the config file (input_size).
(4) Recommend to run the first training with voc0712 dataset as it is widely used and reasonably small.
(5) To convert cityscapes to coco format, run the script: tools/convert_datasets/cityscapes.py

config='./configs/jacinto_ai/ssd_mobilenet.py'
config='./configs/jacinto_ai/ssd_mobilenet_fpn.py'
config='./configs/jacinto_ai/ssd_resnet_fpn.py'
config='./configs/jacinto_ai/ssd_regnet_fpn_bgr.py'

config='./configs/jacinto_ai/retinanet_regnet_fpn_bgr.py'
config='./configs/jacinto_ai/retinanet_resnet_fpn.py'
config='./configs/jacinto_ai/fcos_regnet_fpn_bgr.py'
'''

config='./configs/jacinto_ai/ssd_mobilenet_fpn.py'

########################################################################
# other settings
gpus = 4
dataset_style = 'voc' #'voc' #'coco'
master_port = 29500


########################################################################
# Note: use option to add additional options: eg:
# --option="resume_from=./work_dirs/ssd512_mobilenet_voc/latest.pth"
sys.argv = [sys.argv[0], f'--nproc_per_node={gpus}', f'--master_port={master_port}',
             './tools/train.py', '--launcher=pytorch', config]
distributed_launch.main()



