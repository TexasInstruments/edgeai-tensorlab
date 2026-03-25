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



# using information given in https://huggingface.co/timm/fastvit_s12.apple_in1k
# added reparametrization to export a compact model

!pip3 install onnxsim
!pip3 install onnx==1.13

from urllib.request import urlopen
from PIL import Image
import torch
import timm
import onnxsim
import os

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model_name = 'fastvit_sa24.apple_in1k'
model = timm.create_model(model_name, pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

#top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)

example_input = torch.rand(1,3,256,256)

#torch.onnx.export(model, example_input, f'{model_name}.onnx')
#os.system(f'onnxsim {model_name}.onnx {model_name}_simp.onnx')

for m in model.modules():
  if hasattr(m, 'reparameterize'):
    m.reparameterize()

reparam_model = timm.create_model(model_name, pretrained=False, inference_mode=True)
reparam_model.eval()

state_dict = model.state_dict()
reparam_model.load_state_dict(state_dict)

reparam_output = reparam_model(transforms(img).unsqueeze(0)) 
diff = torch.max(torch.abs(output-reparam_output))
print("max_diff =", diff.item())

torch.onnx.export(model, example_input, f'{model_name}_reparam.onnx')
os.system(f'onnxsim {model_name}_reparam.onnx {model_name}_reparam_simp.onnx')


