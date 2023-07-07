import os
import argparse
import datetime
import torch
import torchvision
import onnx
import onnxsim
from edgeai_torchtoolkit.v2 import xao


def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    # all_models = ["mobilenet_v2", "convnext_large"]
    # all_models = torchvision.models.list_models() # for all models
    all_models = torchvision.models.list_models(module=torchvision.models) # for classification models

    print(all_models)
    input_tensor = torch.rand((1,3,224,224))
    for name in all_models:
        print(f"Exporting {name}")
        model = torchvision.models.get_model(name)
        output_path = os.path.join(args.output_path, name+".onnx")
        torch.onnx.export(model, input_tensor, output_path)
        # surgery
        model = xao.surgery.replace_unsuppoted_layers(model, verbose_mode=False)
        output_path = os.path.join(args.output_path, name+"_lite.onnx")
        torch.onnx.export(model, input_tensor, output_path)
        # onnx_model = onnx.load(output_path)
        # onnx_model, _ = onnxsim.simplify(onnx_model)
        # onnx.save(onnx_model, output_path)

if __name__ == "__main__":
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="PyTorch Model Surgery Examples")
    parser.add_argument("--output-path", default=f"./data/checkpoints/{date}_export", type=str, help="dataset path")
    args = parser.parse_args()
    main(args)
