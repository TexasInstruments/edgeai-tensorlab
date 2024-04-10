
## Quantization methods supported
- MMAv2 devices: All the latest devices (such as AM62A, AM68A/TDA4AL/TDA4VL/TDA4VE, AM69A/TDA4VH AM67A/TDA4AEN) with MMAv2 accelerator supports full non-power-of-2 quantization and per-channel weight quantization. This will provide better accuracy with PTQ. 
- Legacy MMAv1 devices: Legacy TDA4VM device that uses MMAv1 accelerator uses per-tensor, power-of-2-scale quantization which is more restrictive.


## How to quantize models
- TIDL PTQ: On all the supported devices, TIDL supports PTQ using float models. For more details of TIDL PTQ, see the documentation of [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)
- Using edgeai-benchmark wrapper: [edgeai-benchmark](https://github.com/TexasInstruments/edgeai-benchmark) is a wrapper over edgeai-tidl-tools and it is used to compile models in our Model Zoo. The compilation parameters for each model is listed in the [config file published along with the models](https://github.com/TexasInstruments/edgeai-modelzoo/blob/main/models/configs.yaml). This is a yaml representation of [edgeai-benchmark/configs](https://github.com/TexasInstruments/edgeai-benchmark/configs)
- TIDL also supports pre-quantized models, where the pre-quantization can be performed in Pytorch itself. Pre-quantized models can help to get better accuracy or to make the TIDL compilation faster. 


## How to create pre-quantized models

### Native Pytorch quantization 
On all the latest devices with MMAv2 accelerator, TIDL supports models trained with native PyTorch quantization - usage of our QAT/PTQ wrappers below are completely optional for these devices. Read more about **[using Native PyTorch quantized models in TIDL](./native_pytorch_quantization.md)**.

**Note**: Models trained with Native Pytorch quantization is supported in TIDL version 9.1, December 2023 onwards. 

### Latest Quantization Aware Training / QAT (v2) wrapper
[edgeai_torchmodelopt.xmodelopt.quantization.v2](../edgeai_torchmodelopt/xmodelopt/quantization/v2) - Easy to use thin wrapper over Pytorch Native Quantization Aware Training (QAT) that uses **torch.fx** (built on top of **torch.ao** architecture optimization tools). Compared to the legacy QAT using **torch.nn** modules, torch.fx based QAT is able to handle models that uses torch operators and functionals as well. 

Details of this quantization wrapper is available [here](../edgeai_torchmodelopt/xmodelopt/quantization/v2/README.md)<br>

The user guide for the QAT API is documented in [v2 QAT documentation](../edgeai_torchmodelopt/xmodelopt/quantization/v2/docs/qat.md).

**Note**: Models trained with this wrapper is supported in TIDL version 9.1, December 2023 onwards. 

### Legacy Quantization Aware Training / QAT (v1): 
[edgeai_torchmodelopt.xmodelopt.quantization.v1](../edgeai_torchmodelopt/xmodelopt/quantization/v1) - Our legacy implementation of Quantization Aware Training (QAT) using **torch.nn** modules. This is required for QAT that is compatible with TDA4VM (which uses legacy MMAv1 accelerator)<br>

Details of this quantization wrapper is available [here](../edgeai_torchmodelopt/xmodelopt/quantization/v1/README.md)

The user guide for the QAT API is documented in [v1 QAT documentation](../edgeai_torchmodelopt/xmodelopt/quantization/v1/docs/qat.md).

## Results

Following are the results of 8 bit quantization of torchvision models and their lite alternatives. There is a marginal drop in accuracy for these networks. The networks are trained on imagenet dataset using the torchvision training package.

| Models        |  Float Accuracy          | Int8 Quantized Model Accuracy   |
| ------------- |:-------------:    | :-----:                |
| MobileNetv2-lite  | 72.938 | 72.476           |
| ResNet50     | 76.13         |   75.052              |

