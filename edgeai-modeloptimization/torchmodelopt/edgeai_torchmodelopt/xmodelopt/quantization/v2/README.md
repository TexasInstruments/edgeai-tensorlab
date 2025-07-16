
# Introduction

Quantization of a CNN model is the process of converting floating point data & operations to fixed point (integer). This includes quantization of weights, feature maps and all operations (including convolution).

Accuracy of inference can degrade if the CNN model is quantized to 8bits using simple methods and steps have to be taken to minimize this accuracy loss. The parameters of the model need to be adjusted to suit quantization. This includes adjusting of weights, biases and activation ranges. This adjustment can be done as part of Post Training Quantization (PTQ) or as part of Quantization Aware Training (QAT).

## Overview
Inference engines use fixed point arithmetic to implement neural networks efficiently. For example TI Deep Learning Library (TIDL) for TI’s Jacinto7 TDA4x Devices (eg. TDA4VM, TDA4AL, TDA4VL etc) or AM6XA Devices (eg. AM62A, AM68A, AM69A AM67A etc) supports 16-bit and 8-bit fixed point inference modes.

Fixed point mode, especially the 8-bit mode can have accuracy degradation. The tools and guidelines provided here help to avoid accuracy degradation with quantization.

If you are getting accuracy degradation with 8-bit inference, the first thing to check is 16-bit inference. If 16-bit inference provides accuracy close to floating point and 8-bit has an accuracy degradation, there it is likely that the degradation si due to quantization. However, if there is substantial accuracy degradation with 16-bit inference itself, then it is likely that there is some issue other than quantization.

## Background - Pytorch Native Quantization
This module is an easy-to-use wrapper around Pytorch FX mode static quantization. Please read the documentation on [Pytorch FX mode quantization](https://pytorch.org/docs/stable/quantization.html#prototype-maintenance-mode-fx-graph-mode-quantization) and [Pytorch FX static PTQ](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html) to understand more. Please read that document to get a background. To use our wrapper, there is no need to understand all the details there, as we have tried to make the interface as simple as possible.

## Quantization schemes supported in this module
**Post Training Calibration (PTC)**: Post Training Calibration involves range estimation for weights and activations and also minor tweaks to the model (such as bias adjustments). It can be used to ensure that the model is getting properly quantized and the expected accuracy can be evaluated. It is easy to incorporate into an existing PyTorch training code. More details are in [Post Training Calibration(**PTC**) documentation](./docs/ptc.md).

**Quantization Aware Training (QAT)**: This is needed only if the accuracy obtained with PTC is not satisfactory. QAT operates as a second phase after the initial training in floating point, in the training framework. More details are in [Quantization Aware Training (**QAT**) documentation](./docs/qat.md).

**torch.fx friendly code**: It is important to refactor your code such that it is symbolically traceable and properly quantizable (observers are inserted at proper locations). More information about symbolic tracing support can be found [here](https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing). Few common suggestions could be :
1. Removing the assert statements in the code because the symbolic tracing step could have issues.
2. If some module is not supported to be quantized, it can be wrapped with @torch.fx.wrap

## Results

Following are the results of 8 bit quantization of torchvision models and their lite alternatives. There is a marginal drop in accuracy for these networks. The networks are trained on imagenet dataset using the torchvision training package.

| Models        |  Float Accuracy          | Int8 Quantized Model Accuracy   |
| ------------- |:-------------:    | :-----:                |
| MobileNetv2-lite  | 72.938 | 72.476           |
| ResNet50     | 76.13         |   75.052    |

