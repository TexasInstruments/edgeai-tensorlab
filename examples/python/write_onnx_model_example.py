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

import os
import torch
import datetime
from torchvision import models
# from references.edgeailite import models

# dependencies
# Python 3.7 (might work in other versions as well)
# pytorch, torchvision - install using: 
# conda install pytorch torchvision -c pytorch

# some parameters - modify as required
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
dataset_name = 'image_folder_classification'
model_names = ['mobilenet_v2', 'resnet18', 'resnet50', 'resnext50_32x4d', 'shufflenet_v2_x1_0']
img_resize = (256,256)
rand_crop = (224,224)
opset_version = 9

# the saving path - you can choose any path
save_path = './data/checkpoints'
save_path = os.path.join(save_path, dataset_name, date + '_' + dataset_name)
save_path += '_resize{}x{}_traincrop{}x{}'.format(img_resize[1], img_resize[0], rand_crop[1], rand_crop[0])
os.makedirs(save_path, exist_ok=True)

# create a rand input
rand_input = torch.rand(1, 3, rand_crop[0], rand_crop[1])

for model_name in model_names:
    # create the model - replace with your model
    model = models.__dict__[model_name](pretrained=True)
    model.eval()

    # write pytorch model
    model_path=os.path.join(save_path, f'{model_name}_model.pth')
    traced_model = torch.jit.trace(model, rand_input)
    torch.jit.save(traced_model, model_path)

    # write pytorch sate dict
    model_path=os.path.join(save_path, f'{model_name}_state_dict.pth')
    torch.save(model.state_dict(), model_path)

    # write the onnx model
    model_path=os.path.join(save_path, f'{model_name}_opset{opset_version}.onnx')
    torch.onnx.export(model, rand_input, model_path, export_params=True, verbose=False,
                      do_constant_folding=True, opset_version=opset_version)
