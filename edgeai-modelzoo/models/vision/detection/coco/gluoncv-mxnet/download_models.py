# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved
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


# Download MXNet model

import os
import gluoncv


model_details = {
    'yolo3_mobilenet1.0_coco': [{'name': 'data', 'shape': (1, 3, 416, 416)}],
    'ssd_512_mobilenet1.0_coco': [{'name': 'data', 'shape': (1, 3, 512, 512)}],
    'ssd_512_resnet50_v1_coco': [{'name': 'data', 'shape': (1, 3, 512, 512)}],
    'center_net_resnet18_v1b_coco': [{'name': 'data', 'shape': (1, 3, 512, 512)}],
}


for model_name, model_inputs in model_details.items():
    net = gluoncv.model_zoo.get_model(model_name, pretrained=True)
    net_dir = os.path.join('./', model_name)
    gluoncv.utils.export_block(net_dir, net, data_shape=model_inputs[0]['shape'][1:], preprocess=False, layout='CHW')

