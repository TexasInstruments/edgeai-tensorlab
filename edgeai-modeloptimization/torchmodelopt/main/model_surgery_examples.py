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
from edgeai_torchmodelopt import xmodelopt

def main(args):
    if not has_tv:
        print('This script is dependent on torchvision and as it is not installed, the script will close')
        return
    os.makedirs(args.output_path, exist_ok=True)
    all_models = ["mobilenet_v2"]
    # all_models = ["mobilenet_v2", "convnext_large"]
    # all_models = torchvision.models.list_models() # for all models
    # all_models = torchvision.models.list_models(module=torchvision.models) # for classification models

    print(all_models)
    input_tensor = torch.rand((1,3,224,224))
    for name in all_models:
        print(f"Exporting {name}")
        model = torchvision.models.get_model(name)
        output_path = os.path.join(args.output_path, name+".onnx")
        torch.onnx.export(model, input_tensor, output_path, opset_version=18)
        onnx.shape_inference.infer_shapes_path(output_path, output_path)

        # surgery
        if args.model_surgery ==xmodelopt.surgery.SyrgeryVersion.SURGERY_V1:
            model = xmodelopt.surgery.v1.convert_to_lite_model(model)
            name_suffix = "_lite.onnx"
        elif args.model_surgery ==xmodelopt.surgery.SyrgeryVersion.SURGERY_FX:
            model = xmodelopt.surgery.v2.convert_to_lite_fx(model)
            name_suffix = "_lite.onnx"
        else:
            raise RuntimeError('unsupported surgery type')
        #

        # quantization
        # Note: change total_epochs  to the epochs in your training script.
        total_epochs = 1

        if args.quantization:
            if args.quantization == xmodelopt.quantization.QuantizationVersion.QUANTIZATION_LEGACY:
                model = xmodelopt.quantization.v1.QuantTrainModule(model, dummy_input=input_tensor, total_epochs=total_epochs)
                name_suffix = "_quant" + name_suffix
            elif args.quantization == xmodelopt.quantization.QuantizationVersion.QUANTIZATION_FX:
                model = xmodelopt.quantization.v2.QATFxModule(model, qconfig_type=args.quantization_type, total_epochs=total_epochs)
                name_suffix = "_quant" + name_suffix
            else:
                raise RuntimeError('unsupported surgery type')
            #

            # a forward pass is requied for quantization / convert to work
            model(input_tensor)
            model_converted = model.convert() if hasattr(model, 'convert') else model
        else:
            model_converted = model

        output_path = os.path.join(args.output_path, name+name_suffix)
        torch.onnx.export(model_converted, input_tensor, output_path, opset_version=18)
        # onnx_model = onnx.load(output_path)
        # onnx_model, _ = onnxsim.simplify(onnx_model)
        # onnx.save(onnx_model, output_path)
        onnx.shape_inference.infer_shapes_path(output_path, output_path)


if __name__ == "__main__":
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="PyTorch Model Surgery Examples")
    parser.add_argument("--model-surgery", default=xmodelopt.surgery.SyrgeryVersion.SURGERY_FX, type=str, help="model surgery type")
    parser.add_argument("--quantization", "--quantize", dest="quantization", default=0, type=int, choices=xmodelopt.quantization.QuantizationVersion.get_choices(), help="Quaantization Aware Training (QAT)")
    parser.add_argument("--quantization-type", default=None, help="Actual Quaantization Flavour - applies only if quantization is enabled")
    parser.add_argument("--output-path", default=f"./data/checkpoints/{date}_export", type=str, help="dataset path")
    args = parser.parse_args()
    main(args)
