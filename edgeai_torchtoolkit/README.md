# EdgeAI-ModelToolkit / EdgeAI-TorchToolkit

Tools to help development of Embedded Friendly Deep Neural Network Models in [Pytorch](https://pytorch.org).

<hr>

## Categories of tools

### Latest torch.fx based model optimization tools (v2)
- [edgeai_torchtoolkit.v2.xao: eXample model/Architecture Optimization utils](./v2/xao). Includes utilities for Quantization Aware Training (QAT), Model Pruning/Sparsity, Model Surgery etc using **torch.fx** and built on top of **torch.ao** architecture optimization tools.

### Legacy/deprecated torch.nn modules based model optimization tools (v1)
- [edgeai_torchtoolkit.v1.xnn: eXample Neural Network utils](./v1/xnn).  Includes utilities for Quantization Aware Training (QAT) and Model Surgery etc using **torch.nn** modules.

The Quantization and Model Surgery tools are available in both (v1 and v2), but their interfaces and functionality are slighlty different. We recommend to use the latest torch.fx based tools whenever possible, but there may be situations in which legacy tools may be useful. For example the QAT models from torch.fx will be supported in TIDL only from the 9.1 release.

<hr>

# Quantization

There are several tools supported in this repository, and they can be accessed by navigating through the abiove links. However, Quantization  tools require special mention as it is commonly used. The following are the links to the new and old Quantization documentations.

- [Latest torch.fx based Quantization/QAT tools (v2)](./v2/xao/quantization/README.md). Note that For example the QAT models from this will be supported in TIDL only from the 9.1 release.

- [Legacy/deprecated toch.nn modules based Quantization/QAT tools (v1)](./v1/xnn/quantization/README.md)
