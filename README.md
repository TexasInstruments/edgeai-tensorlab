# EdgeAI-ModelToolkit

### Notice
If you have not visited the following landing pages, please do so before attempting to use this repository.
- https://www.ti.com/edgeai 
- https://github.com/TexasInstruments/edgeai

<hr>

## EdgeAI-TorchToolkit introduction
Tools and utilities to help the development of Embedded Friendly Deep Neural Network Models in [Pytorch](https://pytorch.org) - we call these **model optimization tools**. Example tools include:
- Model surgery: replace unsupported layers with supported once - for example SiLU layers can be changed to ReLU layers without changing the model code). Several such transformation are supported and more an be easily added.
- Model pruning: prune/sparsify a model. Both structured and unstructured pruning is supported.
- Quantization Aware training (QAT): easy to use wrapper for Pytorch native QAT with just couple of lines of code change.
- Other utilities: we have other utils functions also, including legacy version of Model surgery and QAT using torch Modules (the newer ones use torch.fx which has superior functionality)

<hr>

### Setup
- [Installation instrunctions](./edgeai_torchtoolkit/docs/setup.md)

<hr>

### Documentation 
- [**EdgeAI-TorchToolkit documentation**](./edgeai_torchtoolkit/README.md) describes how to use these tools. 

<hr>
