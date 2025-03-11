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


import os
import argparse
import datetime
import torch
try:
    import torchvision
    has_tv = True
except:
    has_tv = False
import onnx
import onnxsim


def main(args):
    if not has_tv:
        print('This script is dependent on torchvision and as it is not installed, the script will close')
        return
    os.makedirs(args.output_path, exist_ok=True)
    # all_models = ["mobilenet_v2"]
    # all_models = ["mobilenet_v2", "convnext_large"]
    # all_models = torchvision.models.list_models() # for all models
    all_models = torchvision.models.list_models(module=torchvision.models) # for classification models

    print(all_models)
    input_tensor = torch.rand((1,3,224,224))
    for name in all_models:
        print(f"Checking {name}")
        output_path = os.path.join(args.output_path, name+".onnx")
        if not os.path.exists(output_path):
            try:
                print(f"Exporting {name}")
                model = torchvision.models.get_model(name)
                model.eval()
                torch.onnx.export(model, input_tensor, output_path, opset_version=18)
                onnx.shape_inference.infer_shapes_path(output_path, output_path)
            except:
                print(f"error occurred exporting: {output_path}")


if __name__ == "__main__":
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="PyTorch Model Export Examples")
    parser.add_argument("--output-path", default=f"./data/checkpoints/export", type=str, help="dataset path")
    args = parser.parse_args()
    main(args)
