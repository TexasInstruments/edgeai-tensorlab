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

from urllib.request import urlopen
from PIL import Image
import timm
import onnxsim


def export_model(model_name):
    output_folder = './exports/timm'

    img = Image.open(urlopen(
        'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    ))

    model = timm.create_model(model_name, pretrained=True)
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    input_tensor = transforms(img).unsqueeze(0)

    output = model(input_tensor)  # unsqueeze single image into batch of 1
    top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

    os.makedirs(output_folder, exist_ok=True)
    onnx_file = os.path.join(output_folder, f'{model_name}.onnx')

    torch.onnx.export(model, input_tensor, onnx_file)
    os.system(f'onnxsim {onnx_file} {onnx_file}')


def main():
    # export_model('efficientvit_m5.r224_in1k')
    # export_model('efficientvit_b0.r224_in1k')
    # export_model('efficientvit_b1.r224_in1k')
    # export_model('efficientvit_b2.r224_in1k')
    # export_model('efficientvit_l1.r224_in1k')
    # export_model('efficientvit_l2.r224_in1k')
    export_model('deit_small_patch16_224')


if __name__ == '__main__':
    main()