
Note: Some information in the [Quantization landing page](./quantization.md) may he helpful understanding the background.

## Introduction

The introduction to [Native Pytorch Quantization is available at](https://pytorch.org/docs/stable/quantization.html)

That documentation is lengthy and has a lot of details. Here are the relevant points to consider:
- Eager model quantization and FX Graph Mode quantization supports similar quantization modes, but FX mode is more flexible and is able to handle more variety of models that includes functionals torch.add, torch.relu or operators such as +
- *There is a newer quantization model called PyTorch 2 Export Quantization - that is not yet supported in TIDL.*
- Static quantization is to be used, and dynamic quantization is not supported in TIDL.
- Either PTQ or QAT can be used, both of them produces the quantized model in similar format. QAT can give better accuracy, but can be more involved.

## Pytorch Quantization types

### PTQ with FX Graph mode static quantization

[Docuemtation for PTQ with FX Graph mode static quantization](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html)

```
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping

float_model.eval()
qconfig = get_default_qconfig("qnnpack")
qconfig_mapping = QConfigMapping().set_global(qconfig)
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)


example_inputs = (next(iter(data_loader))[0]) # get an example input

# example_inputs is optional and can be None for prepare_fx
prepared_model = prepare_fx(float_model, qconfig_mapping, example_inputs)  # fuse modules and insert observers

calibrate(prepared_model, data_loader)  # run calibration on sample data

quantized_model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model

# export a quantized onnx model in QDQ format
torch.onnx.export(quantized_model, example_inputs, "model_qdq.onnx")
```

## QAT with FX Graph mode static quantization

```
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization import QConfigMapping

float_model.eval()
qconfig = get_default_qconfig("qnnpack")
qconfig_mapping = QConfigMapping().set_global(qconfig)

example_inputs = (next(iter(data_loader_train))[0]) # get an example input

# example_inputs is optional and can be None for 
prepared_model = prepare_qat_fx(float_model, qconfig_mapping, example_inputs)  # fuse modules and insert observers

train(prepared_model, data_loader_train, data_loader_test)  # run training

quantized_model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model

# export a quantized onnx model in QDQ format
torch.onnx.export(quantized_model, example_inputs, "model_qdq.onnx")
```

## Compiling a pre-quantized model in TIDL

A quantized (QDQ) onnx model can be used in TIDL by setting the flag in the runtime options that is passed for compilation.

```
advanced_options:prequantized_model: 1
```
