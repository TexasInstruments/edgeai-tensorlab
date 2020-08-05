#!/usr/bin/env python

import sys
import os
import datetime
from torch.distributed import launch as distributed_launch

########################################################################
# config
from detection_configs import config

########################################################################
# other settings
distributed = 1
gpus = 4
dataset_style = 'coco' #'voc' #'coco'
master_port = 29500

########################################################################
base_path = os.path.splitext(os.path.basename(config))[0]
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print(f'Training with: {config} @ {date}')

if distributed:
    sys.argv = [sys.argv[0], f'--nproc_per_node={gpus}', f'--master_port={master_port}',
                './xmmdet/tools/train.py', '--launcher=pytorch',
                config]

    distributed_launch.main()
else:
    from xmmdet.tools import train as train_mmdet
    sys.argv = [sys.argv[0], f'--gpus={gpus}', '--no-validate',
                f'{config}']

    args = train_mmdet.parse_args()
    train_mmdet.main(args)
#


