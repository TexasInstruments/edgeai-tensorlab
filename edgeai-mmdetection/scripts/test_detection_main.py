#!/usr/bin/env python

# Copyright (c) 2018-2020, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
master_port = 29501

########################################################################
metric = ('bbox' if dataset_style=='coco' else 'mAP')

base_path = os.path.splitext(os.path.basename(config))[0]
work_dir = os.path.join('./work_dirs', base_path)
checkpoint = f'{work_dir}/latest.pth'
outfile = os.path.join(work_dir,'result.pkl')
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print(f'Testing with: {config} @ {date}')

if distributed:
    sys.argv = [sys.argv[0], f'--nproc_per_node={gpus}', f'--master_port={master_port}',
                './xmmdet/tools/test.py', f'--eval={metric}',
                f'--out={outfile}', '--launcher=pytorch',
                config, checkpoint]
    distributed_launch.main()
else:
    from xmmdet.tools import test as test_mmdet
    sys.argv = [sys.argv[0], f'--eval={metric}',
                f'--out={outfile}',
                f'{config}', f'{checkpoint}']
    args = test_mmdet.parse_args()
    test_mmdet.main(args)
#
