
## Quantization methods supported
- MMAv2 devices: All the latest devices (such as AM62A, AM68A/TDA4AL/TDA4VL/TDA4VE, AM69A/TDA4VH AM67A/TDA4AEN) with MMAv2 accelerator supports full non-power-of-2 quantization and per-channel weight quantization. This will provide better accuracy with PTQ. 
- Legacy MMAv1 devices: Legacy TDA4VM device that uses MMAv1 accelerator uses per-tensor, power-of-2-scale quantization which is more restrictive.


## How to quantize models
- Option 1: TIDL Post Training Quantization (PTQ): On all the supported devices, TIDL supports PTQ using float models. For more details of TIDL PTQ, see the documentation of [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools). Using edgeai-benchmark wrapper: [edgeai-benchmark](https://github.com/TexasInstruments/edgeai-benchmark) is a wrapper over edgeai-tidl-tools and it is used to compile models in our Model Zoo. The compilation parameters for each model is listed in the [config file published along with the models](https://github.com/TexasInstruments/edgeai-modelzoo/blob/main/models/configs.yaml). This is a yaml representation of [edgeai-benchmark/configs](https://github.com/TexasInstruments/edgeai-benchmark/configs)
- Option 2: This repository can be used to create pre-quantized models using Quantization Aware Training (QAT) or Post Training Calibation (PTC). Pre-quantized models can help to get better accuracy or to make the TIDL compilation faster. The pre-quantized models created using this repository can be imported in TIDL.


## How to create pre-quantized models using this repository

### Latest Quantization Aware Training / QAT (v2) wrapper
[edgeai_torchmodelopt.xmodelopt.quantization.v2](./v2) - Easy to use thin wrapper over Pytorch Native Quantization Aware Training (QAT) that uses **torch.fx** (built on top of **torch.ao** architecture optimization tools). Compared to the legacy QAT using **torch.nn** modules, torch.fx based QAT is able to handle models that uses torch operators and functionals as well. 

Details of this quantization wrapper is available [here](./v2/README.md)<br>

**Note**: Models trained with this wrapper is supported in TIDL version 9.1, December 2023 onwards. 

### Legacy Quantization Aware Training / QAT (v1): 
[edgeai_torchmodelopt.xmodelopt.quantization.v1](./v1) - Our legacy implementation of Quantization Aware Training (QAT) using **torch.nn** modules. This is required for QAT that is compatible with TDA4VM (which uses legacy MMAv1 accelerator)<br>

Details of this quantization wrapper is available [here](./v1/README.md)

