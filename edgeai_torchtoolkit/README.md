# EdgeAI-ModelToolkit / EdgeAI-TorchToolkit

Tools to help development of Embedded Friendly Deep Neural Network Models in [Pytorch](https://pytorch.org) we call these **model optimization tools**.
- [xao](xao) contains our extension of [torch.ao](https://github.com/pytorch/pytorch/tree/main/torch/ao) model architecture optimization tools.
- [xnn](xnn) contains our extension of [torch.nn](https://github.com/pytorch/pytorch/tree/main/torch/nn) neural network model utilities.


<hr>

## Quantization

We recommend to use the latest version (v2) tools whenever possible, but there may be situations in which legacy tools may be useful. For example the QAT models from torch.fx will be supported in TIDL only from the 9.1 release onwards.

### **Latest** Quantization Aware Training / QAT (v2)
- [xao/quantization/v2](./xao/quantization/v2) - Easy to use wrapper over Pytorch native Quantization Aware Training (QAT) that uses **torch.fx** (built on top of **torch.ao** architecture optimization tools). Compared to the legacy QAT using **torch.nn** modules, torch.fx based QAT is able to handle models that uses torch operators and functionals as well.
- **Note**: Models trained with this v2 QAT tool is supported in TIDL version 9.1 onwards (November 2023) 

### Legacy Quantization Aware Training / QAT (v1)
- [xao/quantization/v1](./xao/quantization/v1) - Our legacy implementation of Quantization Aware Training (QAT) using **torch.nn** modules.


## Model surgery

### **Latest** Model surgery tool (v2)
- [xao/surgery/v2](./xao/surgery/v2) - Easily replace layers (Modules, operators, functional) with other layers without modifying the model code. Easily replace unsupported layers (in a certain SoC) with other supported layers; create embedded friendly models. Compared to the legacy surgery using **torch.nn** modules, torch.fx based surgery is able to handle models that uses torch operators and functionals as well.

### Legacy Model surgery tool (v1)
- [xao/surgery/v1](./xao/surgery/v1) - Our legacy implementation of Model surgery using **torch.nn** modules. 

## Model pruning
- [xao/pruning](./xao/pruning) - prune/sparsify a model. Both structured and unstructured pruning is supported.


<hr>


## Note
The Quantization and Model Surgery tools are available in both (v1 and v2), but their interfaces and functionality are slightly different. We recommend to use the latest version (torch.fx based v2) tools whenever possible, but there may be situations in which legacy tools may be useful. For example the QAT models from torch.fx will be supported in TIDL only from the 9.1 release.