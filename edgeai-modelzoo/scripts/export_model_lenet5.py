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
import torchvision
import onnx
import torch.nn as nn
import torch.nn.functional as F

output_folder = './exports/lenet5'


# https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/2_lenet.ipynb
class LeNet5(nn.Module):
    def __init__(self, output_dim=10):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=5)

        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)

        self.fc_1 = nn.Linear(16 * 4 * 4, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, output_dim)

    def forward(self, x):
        # x = [batch size, 1, 28, 28]
        x = self.conv1(x)
        # x = [batch size, 6, 24, 24]
        x = F.max_pool2d(x, kernel_size=2)
        # x = [batch size, 6, 12, 12]
        x = F.relu(x)
        x = self.conv2(x)
        # x = [batch size, 16, 8, 8]
        x = F.max_pool2d(x, kernel_size=2)
        # x = [batch size, 16, 4, 4]
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        # x = [batch size, 16*4*4 = 256]
        h = x
        x = self.fc_1(x)
        # x = [batch size, 120]
        x = F.relu(x)
        x = self.fc_2(x)
        # x = batch size, 84]
        x = F.relu(x)
        x = self.fc_3(x)
        # x = [batch size, output dim]
        return x



model_name = 'lenet5_untrained'
input_size = (28,28)
input_tensor = torch.rand(1,1,*input_size)

model = LeNet5()
model.eval()

os.makedirs(output_folder, exist_ok=True)
onnx_file = os.path.join(output_folder, f'{model_name}.onnx')

torch.onnx.export(model, input_tensor, onnx_file, opset_version=17)

#onnx.shape_inference.infer_shapes_path(onnx_file, onnx_file)
os.system(f"onnxsim {onnx_file} {onnx_file}")
