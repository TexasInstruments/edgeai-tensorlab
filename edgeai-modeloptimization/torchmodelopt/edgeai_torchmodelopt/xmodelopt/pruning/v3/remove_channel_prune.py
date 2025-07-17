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
import torch._dynamo as torch_dynamo
import torch.fx as fx
try:
    import torchvision
    has_tv = True
except:
    has_tv = False
from edgeai_torchmodelopt.xmodelopt.pruning.v3.utils import create_channel_pruned_model,register_custom_ops_for_onnx
import copy


def main():
    if not has_tv:
        print('''
    This is a test script for channel pruning. This only supports torchvision till now. So, the code will exit now.
    ''')
        return
    num_classes = 10
    torch.backends.cuda.enable_flash_sdp(False)

    model = torchvision.models.vit_b_16(num_classes =num_classes)
    # model = torchvision.models.resnet50()
    dummy_input = torch.randn(10, 3, 224, 224)
    m,_ = torch_dynamo.export(model,aten_graph=True,pre_dispatch=True,assume_static_by_default=True)(dummy_input)

    current_model_dict = m.state_dict()
    model_path = '/home/a0507161/Kunal/transformer_sparsity/outputs/vit_b_16/2024_06_12_16_52_07/last_checkpoint.pth'
    state_dict = torch.load(model_path)
    state_dict = state_dict['model']

    new_state_dict={}
    for k,v in state_dict.items():
        param_names = k.split('.')
        new_state_dict[param_names[-1] ] = v
    m.load_state_dict(state_dict=new_state_dict)

    orig_model = copy.deepcopy(m)

    final_model = create_channel_pruned_model(m)
    # print('\n'.join([str((n,p.shape)) for n,p in final_model.named_parameters()]))

    print("The forward pass is starting \n")

    y = final_model(dummy_input)
    print("The forward pass completed \n")
    register_custom_ops_for_onnx(17)
    torch.onnx.export(orig_model, dummy_input, model_path[:-4]+"_orig.onnx")
    torch.onnx.export(final_model, dummy_input, model_path[:-4]+"_final.onnx")

if __name__ == '__main__':
    main()