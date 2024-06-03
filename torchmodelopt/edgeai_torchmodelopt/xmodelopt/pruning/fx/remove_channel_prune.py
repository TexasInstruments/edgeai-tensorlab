#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################


import torch
import torch.fx as fx
import torchvision
from edgeai_torchmodelopt.xmodelopt.pruning import create_channel_pruned_model2
import copy

num_classes = 10
model = torchvision.models.vit_b_16(num_classes =num_classes)
# model = torchvision.models.resnet50()
current_model_dict = model.state_dict()
model_path = '/home/a0507161/Kunal/transformer_sparsity/outputs/vit_b_16/2024_05_21_17_58_52/last_checkpoint.pth'
state_dict = torch.load(model_path)
state_dict = state_dict['model']

new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), state_dict.values())}
model.load_state_dict(state_dict=new_state_dict)

orig_model = copy.deepcopy(model)

final_model = create_channel_pruned_model2(model)

dummy_input = torch.randn(10, 3, 224, 224)
print("The forward pass is starting \n")

y = final_model(dummy_input)
print("The forward pass completed \n")

torch.onnx.export(orig_model, dummy_input, model_path[:-4]+"_orig.onnx")
torch.onnx.export(final_model, dummy_input, model_path[:-4]+"_final.onnx")