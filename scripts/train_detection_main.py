#!/usr/bin/env python

import sys
import os
import datetime
from torch.distributed import launch as distributed_launch

########################################################################
# config

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
config='./configs/retinanet/retinanet-lite_regnet_bifpn_bgr.py'
config='./configs/retinanet/retinanet-lite-xtra-aug_regnet_bifpn_bgr.py'

config='./configs/fcos/fcos-lite-xtra-aug_regnet_bifpn_bgr.py'
config='./configs/fcos/fcos-lite_regnet_bifpn_bgr.py'
'''

config='./configs/retinanet/retinanet-lite-xtra-aug_regnet_bifpn_bgr.py'

########################################################################
# other settings
distributed = 1
gpus = 4
dataset_style = 'coco' #'voc' #'coco'
master_port = 29500

########################################################################
base_path = os.path.splitext(os.path.basename(config))[0]
work_dir = os.path.join('./data/checkpoints/object_detection', base_path)
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print(f'Saving to: {work_dir} @ {date}')

if distributed:
    sys.argv = [sys.argv[0], f'--nproc_per_node={gpus}', f'--master_port={master_port}',
                './xmmdet/tools/train.py', '--launcher=pytorch',
                f'--option=work_dir={work_dir}', config]

    distributed_launch.main()
else:
    from xmmdet.tools import train as train_mmdet
    sys.argv = [sys.argv[0], f'--gpus={gpus}', '--no-validate',
                f'--option=work_dir={work_dir}', f'{config}']

    args = train_mmdet.parse_args()
    train_mmdet.main(args)
#


