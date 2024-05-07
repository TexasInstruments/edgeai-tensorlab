
This module is an easy-to-use wrapper around [Pytorch native QAT](https://pytorch.org/docs/stable/quantization.html). Please read that document to get a background. To use our wrapper, there is no need to understand all the details there, as we have tried to make the interface as simple as possible.

# Introduction

Quantization of a CNN model is the process of converting floating point data & operations to fixed point (integer). This includes quantization of weights, feature maps and all operations (including convolution).

Accuracy of inference can degrade if the CNN model is quantized to 8bits using simple methods and steps have to be taken to minimize this accuracy loss. The parameters of the model need to be adjusted to suit quantization. This includes adjusting of weights, biases and activation ranges. This adjustment can be done as part of Post Training Quantization (PTQ) or as part of Quantization Aware Training (QAT).


## Overview
Inference engines use fixed point arithmetic to implement neural networks efficiently. For example TI Deep Learning Library (TIDL) for TI’s Jacinto7 TDA4x Devices (eg. TDA4VM, TDA4AL, TDA4VL etc) or AM6XA Devices (eg. AM62A, AM68A, AM69A AM67A etc) supports 16-bit and 8-bit fixed point inference modes.

Fixed point mode, especially the 8-bit mode can have accuracy degradation. The tools and guidelines provided here help to avoid accuracy degradation with quantization.

If you are getting accuracy degradation with 8-bit inference, the first thing to check is 16-bit inference. If 16-bit inference provides accuracy close to floating point and 8-bit has an accuracy degradation, there it is likely that the degradation si due to quantization. However, if there is substantial accuracy degradation with 16-bit inference itself, then it is likely that there is some issue other than quantization.

### Quantization Schemes
In this repository, we have  [guidelines](./docs/guidelines.md) on how to choose models and how train them to get the best accuracy with Quantization. It is unlikely that there will be significant accuracy drop with PTQ if these guidelines are followed. In spite of this, if there are models that have significant accuracy drop with quantization, it is possible to improve the accuracy using QAT.

Post Training Quantization (PTQ): Post Training Quantization involves range estimation for weights and activations and also minor tweaks to the model (such as bias adjustments). TIDL can accept a floating point model and do PTQ using a few sample images. This is done during the import of the model in TIDL. More details are in [Post Training Quantization in TIDL (**PTQ**) documentation](./docs/ptq.md).

Quantization Aware Training (QAT): This is needed only if the accuracy obtained with PTQ is not satisfactory. QAT operates as a second phase after the initial training in floating point, in the training framework. More details are in [Quantization Aware Training (**QAT**) documentation](./docs/qat.md).


