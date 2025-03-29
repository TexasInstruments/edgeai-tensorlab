
## Quantization methods supported
* MMAv2 devices: All the latest devices (such as AM62A, AM68A/TDA4AL/TDA4VL/TDA4VE, AM69A/TDA4VH AM67A/TDA4AEN) with MMAv2 accelerator supports full non-power-of-2 quantization and per-channel weight quantization. This will provide better accuracy with PTQ. 

* Legacy MMAv1 devices: Legacy TDA4VM device that uses MMAv1 accelerator uses per-tensor, power-of-2-scale quantization which is more restrictive.


## How to quantize models
### Option 1: TIDL Post Training Quantization (**PTQ**)
* On all the supported devices, TIDL supports PTQ using float models. For more details of TIDL PTQ, see the documentation of [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) and some additional notes are available in [PTQ usage for TIDL](./v1/docs/tidl_ptq.md). 
* Using edgeai-benchmark wrapper: [edgeai-benchmark](https://github.com/TexasInstruments/edgeai-benchmark) is a wrapper over edgeai-tidl-tools and it is used to compile models in our Model Zoo. The compilation parameters for each model is listed in the [config file published along with the models](https://github.com/TexasInstruments/edgeai-modelzoo/blob/main/models/configs.yaml). This is a yaml representation of [edgeai-benchmark/configs](https://github.com/TexasInstruments/edgeai-benchmark/configs).

### Option 2: Quantization Aware Training (**QAT**) & Post Training Calibration ((**PTC**) using this repository
* QAT is typically introduced into the training loop and actively learns the quantization parameters and the model weights. However, PTC is a quick and easy way to find the quantization parameters without using backpropagation or training loop.

* This repository uses Pytorch Quantization tools for QAT and PTC. This repository can be used to create pre-quantized models which can then be imported in TIDL. Pre-quantized models can help to get better accuracy or to make the TIDL compilation faster. [Pytorch Quantization documentation](https://pytorch.org/docs/stable/quantization.html) may be helpful to understand details of the quantization tools supported in Pytorch.


## Important Guidelines
In this repository, we have  [**guidelines**](./docs/guidelines.md) on how to choose models and how train them to get the best accuracy with Quantization. It is unlikely that there will be significant accuracy drop with PTQ if these guidelines are followed. In spite of this, if there are models that have significant accuracy drop with quantization, it is possible to improve the accuracy using QAT.


## How to create pre-quantized models using this repository

### [Latest Quantization wrapper (v2)](./v2)
Easy to use wrapper over Pytorch Native Quantization that uses **torch.fx** (built on top of **torch.ao** architecture optimization tools. Please see the [**documentation here**](./v2/README.md). 

Compared to the legacy QAT using **torch.nn** modules, torch.fx based QAT is able to handle models that uses torch operators and functionals as well. 

Details of this Quantization Aware Training (QAT) wrapper is [here](./v2/docs/qat.md)<br>

Details of this Post Training Calibration (PTC) wrapper is [here](./v2/docs/ptc.md)<br>

**Note**: Models trained with these wrapper is supported in TIDL version 9.1, December 2023 onwards. 


### [Legacy Quantization wrapper (v1)](./v1)
Our legacy implementation of Quantization Aware Training (QAT) using **torch.nn** modules. This is required for QAT that is compatible with TDA4VM (which uses legacy MMAv1 accelerator)<br>

Details of this quantization wrapper is available [**here**](./v1/README.md)

