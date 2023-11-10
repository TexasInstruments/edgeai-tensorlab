# EdgeAI-ModelToolkit / EdgeAI-TorchToolkit

Tools to help development of Embedded Friendly Deep Neural Network Models in [Pytorch](https://pytorch.org) we call these **model optimization tools**. This contains [xmodelopt](xmodelopt) which is our extension of [torch.ao](https://github.com/pytorch/pytorch/tree/main/torch/ao) model architecture optimization tools. These are tools to make models more efficient, embedded friendly and replace any layers that are not supported in TIDL with other supported layers.<br>

<hr>

## Quantization
- **Latest Quantization Aware Training / QAT (v2): [edgeai_torchtoolkit.xmodelopt.quantization.v2](./xmodelopt/quantization/v2)** - Easy to use wrapper over Pytorch native Quantization Aware Training (QAT) that uses **torch.fx** (built on top of **torch.ao** architecture optimization tools). Compared to the legacy QAT using **torch.nn** modules, torch.fx based QAT is able to handle models that uses torch operators and functionals as well. (**Note**: Models trained with this v2 QAT tool is supported in TIDL version 9.1  November 2023 onwards)<br><br>

- Legacy Quantization Aware Training / QAT (v1): [edgeai_torchtoolkit.xmodelopt.quantization.v1](./xmodelopt/quantization/v1) - Our legacy implementation of Quantization Aware Training (QAT) using **torch.nn** modules.<br>

<hr>

## Model surgery
- **Latest Model surgery tool (v2)**: [edgeai_torchtoolkit.xmodelopt.surgery.v2](./xmodelopt/surgery/v2) - Easily replace layers (Modules, operators, functional) with other layers without modifying the model code. Easily replace unsupported layers (in a certain SoC) with other supported layers; create embedded friendly models. This uses torch.fx based surgery to handle models that uses torch modules, operators and functionals. (Compared to this, legacy surgery using **torch.nn** can only handle modules)<br><br>

- Legacy Model surgery tool (v1): [edgeai_torchtoolkit.xmodelopt.surgery.v1](./xmodelopt/surgery/v1) - Our legacy implementation of Model surgery using **torch.nn** modules.<br>

<hr>

## Model pruning
- Prune/sparsify a model: [edgeai_torchtoolkit.xmodelopt.pruning](./xmodelopt/pruning) - Both structured and unstructured pruning is supported. Structured pruning includes N:M pruning and channel pruning.

<hr>

## Note
- The Quantization and Model Surgery tools are available in both (v1 and v2), but their interfaces and functionality are slightly different. We recommend to use the latest version (torch.fx based v2) tools whenever possible, but there may be situations in which legacy tools may be useful. For example the QAT models from torch.fx will be supported in TIDL only from the 9.1 release (November 2023).<br><br>

- This package also contains [edgeai_torchtoolkit.xnn](xnn) which is our extension of [torch.nn](https://github.com/pytorch/pytorch/tree/main/torch/nn) neural network model utilities. This is for advanced users and contains several undocumented utilities. <br>