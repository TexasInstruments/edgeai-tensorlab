#!/usr/bin/env python

import sys
import os
from tools import test as test_mmdet

########################################################################
# config

'''
Usage: 
(1) Use one of the following config files.
(2) Inside the config file, make sure that the dataset that needs to be trained on is uncommented.
(3) Use the appropriate input resolution in the config file (input_size).
(4) Recommend to run the first training with voc0712 dataset as it is widely used and reasonably small.
(5) To convert cityscapes to coco format, run the script: tools/convert_datasets/cityscapes.py

config='./configs/ssd/ssd_mobilenet.py'
config='./configs/ssd/ssd_mobilenet_fpn.py'
config='./configs/ssd/ssd_resnet_fpn.py'
config='./configs/ssd/ssd_regnet_fpn_bgr.py'

config='./configs/retinanet/retinanet_resnet_fpn.py'
config='./configs/retinanet/retinanet_regnet_fpn_bgr.py'
config='./configs/retinanet/retinanet_regnet_bifpn_bgr.py'
config='./configs/retinanet/fcos_regnet_fpn_bgr.py'
'''

config='./configs/retinanet/retinanet_regnet_fpn_bgr.py'

########################################################################
# other settings
gpus = 4
dataset_style = 'coco' #'voc' #'coco'


#########################################################################
metric = ('bbox' if dataset_style=='coco' else 'mAP')
basename = os.path.splitext(os.path.basename(config))[0]
outdir = os.path.join('./work_dirs', basename)
checkpoint = f'{outdir}/latest.pth'
outfile = os.path.join(outdir,'result.pkl')

print(f'Saving to: {outdir}')
sys.argv = [sys.argv[0], f'--out={outfile}', f'--eval={metric}', f'{config}', f'{checkpoint}']
args = test_mmdet.parse_args()
test_mmdet.main(args)
