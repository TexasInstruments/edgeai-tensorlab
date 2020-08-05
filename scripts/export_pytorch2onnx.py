#!/usr/bin/env python

import sys
import os
import datetime

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
config='./configs/ssd/ssd-lite_regnet_bifpn_bgr.py'

config='./configs/retinanet/retinanet_resnet_fpn.py'
config='./configs/retinanet/retinanet-lite_regnet_fpn_bgr.py'
config='./configs/retinanet/retinanet-lite_regnet_bifpn_bgr.py'
'''

config='./configs/ssd/ssd-lite_regnet_fpn_bgr.py'

########################################################################


########################################################################

base_path = os.path.splitext(os.path.basename(config))[0]
work_dir = os.path.join('./work_dirs', base_path)
checkpoint = f'{work_dir}/latest.pth'
outfile = os.path.join(work_dir,'model.onnx')
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
shape = [512, 512]

print(f'Exporting ONNX with: {config} @ {date}')

from xmmdet.tools import pytorch2onnx
sys.argv = [sys.argv[0], f'{config}', f'{checkpoint}', f'--out={outfile}', f'--shape', f'{shape[0]}', f'{shape[1]}']
args = pytorch2onnx.parse_args()
pytorch2onnx.main(args)

