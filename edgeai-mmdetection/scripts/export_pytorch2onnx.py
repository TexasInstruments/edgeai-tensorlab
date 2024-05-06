#!/usr/bin/env python

# Copyright (c) 2018-2021, Texas Instruments
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
import mmcv
from tools.deployment import pytorch2onnx

# config from local folder
from detection_configs import config

########################################################################
base_path = os.path.splitext(os.path.basename(config))[0]
work_dir = os.path.join('./work_dirs', base_path)
checkpoint = f'{work_dir}/latest.pth'
outfile = os.path.join(work_dir,'model.onnx')
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
opset_version = 11

cfg = mmcv.Config.fromfile(config)
input_size = cfg.input_size if hasattr(cfg, 'input_size') else None #(512,512)
assert input_size is not None, 'input_size could not be obtained automatically - please populate manually'

print(f'Exporting onnx+proto model for: {config} @ {date} with input_size {input_size}')
sys.argv = [sys.argv[0], f'{config}', f'{checkpoint}', f'--shape', f'{input_size[0]}', f'{input_size[1]}',
            f'--opset-version', f'{opset_version}',  f'--out={outfile}']
args = pytorch2onnx.parse_args()
pytorch2onnx.main(args)

