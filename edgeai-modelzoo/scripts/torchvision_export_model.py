import torch
import torchvision

models_list = dir(torchvision.models)
models_list = [m for m in models_list if not m.startswith('__')]
print(models_list)

model_name = 'resnet34'
input_size = (224,224)
input_tensor = torch.rand(1,3,*input_size)

model_creator = torchvision.models.__dict__[model_name]
model = model_creator()
model.eval()

onnx_file = f'{model_name}.onnx'

torch.onnx.export(model, input_tensor, onnx_file, opset_version=11)

