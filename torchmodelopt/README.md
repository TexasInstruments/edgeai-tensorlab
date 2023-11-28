# EdgeAI-ModelToolkit / EdgeAI-TorchToolkit

## Introduction
Tools to help development of Embedded Friendly Deep Neural Network Models in [Pytorch](https://pytorch.org) we call these **model optimization tools**. This contains [xmodelopt](xmodelopt) which is our extension of [torch.ao](https://github.com/pytorch/pytorch/tree/main/torch/ao) model architecture optimization tools. These are tools to make models more efficient and embedded friendly. <br>

## Overview
Torch model optimiation toolkit supports the following major features.

### Quantization:

- **Latest Quantization Aware Training / QAT (v2): [edgeai_torchtoolkit.xmodelopt.quantization.v2](./xmodelopt/quantization/v2)** - Easy to use wrapper over Pytorch native Quantization Aware Training (QAT) that uses **torch.fx** (built on top of **torch.ao** architecture optimization tools). Compared to the legacy QAT using **torch.nn** modules, torch.fx based QAT is able to handle models that uses torch operators and functionals as well. (**Note**: Models trained with this v2 QAT tool is supported in TIDL version 9.1  November 2023 onwards)<br>

- Legacy Quantization Aware Training / QAT (v1): [edgeai_torchtoolkit.xmodelopt.quantization.v1](./xmodelopt/quantization/v1) - Our legacy implementation of Quantization Aware Training (QAT) using **torch.nn** modules.<br>

### Model Surgery:

- **Latest Model surgery tool (v2)**: [edgeai_torchtoolkit.xmodelopt.surgery.v2](./xmodelopt/surgery/v2) - Easily replace layers (Modules, operators, functional) with other layers without modifying the model code. Easily replace unsupported layers (in a certain SoC) with other supported layers; create embedded friendly models. This uses torch.fx based surgery to handle models that uses torch modules, operators and functionals. (Compared to this, legacy surgery using **torch.nn** can only handle modules)<br>

- Legacy Model surgery tool (v1): [edgeai_torchtoolkit.xmodelopt.surgery.v1](./xmodelopt/surgery/v1) - Our legacy implementation of Model surgery using **torch.nn** modules.<br>

### Sparsity:

- Prune/sparsify a model: [edgeai_torchtoolkit.xmodelopt.pruning](./xmodelopt/pruning) - Both structured and unstructured pruning is supported. Structured pruning includes N:M pruning and channel pruning. Here, we provide with a few parametrization techniques, and the user can bring their own technique as well and implement as a part of the toolkit.
<br>

- Channel Sparsity uses torch.fx to find the connections and obtain a smaller network after the training with induced sparsity finishes. 

### Model Utilities:

- This package also contains edgeai_torchtoolkit.xnn which is our extension of torch.nn neural network model utilities. This is for advanced users and contains several undocumented utilities.

> ## Note
> The Quantization and Model Surgery tools are available in both (v1 and v2), but their interfaces and functionality are slightly different. We recommend to use the latest version (torch.fx based v2) tools whenever possible, but there may be situations in which legacy tools may be useful. For example the QAT models from torch.fx will be supported in TIDL only from the 9.1 release (November 2023). <br> <br>

<!-- 

### Supported Devices

### Why use our toolkit?

Our toolkit provides the APIs for quantization, surgery as well as sparsity along with multiple torch.nn tools, for user to seemlessly introduce them in their own training code. 
The user can add a single line of code to introduce each of them as shown in the user guides. 

-->


# Getting Started

## Installation

### pip installation
> pip3 install "git+https://github.com/TexasInstruments/edgeai-modeloptimization.git#subdirectory=torchmodelopt"

### Local (editable) Installation
Install this repository as a local editable Python package (for development) by running setup.sh from the directory torchmodelopt:

> ./setup.sh


## User Guides

### Quantization

    import edgeai_torchtoolkit
    # wrap your model in xnn.quantization.QATFxModule. 
    # once it is wrapped, the actual model is in model.module
    model = edgeai_torchtoolkit.xmodelopt.quantization.v2.QATFxModule(model)

This is the basic usage, the detailed usage for the API is documented in [Quantization](/edgeai-modeloptimization/torchmodelopt/edgeai_torchmodelopt/xmodelopt/quantization/v2/README.md).

### Model Surgery

    # Using the default replacement dictionary in surgery
    import edgeai_torchtoolkit
    model = edgeai_torchtoolkit.xmodelopt.surgery.v2.convert_to_lite_fx(model)

    # adding custom layers for replacement in surgery
    import edgeai_torchtoolkit
    replacement_dict = edgeai_torchtoolkit.xmodelopt.surgery.v2.get_replacement_dict_default()
    replacement_dict.update({'layerNorm':custom_surgery_functions.replace_layer_norm})
    model = edgeai_torchtoolkit.xmodelopt.surgery.v2.convert_to_lite_fx(model, replacement_dict=replacement_dict)

This is the basic usage, the detailed usage for the API is documented in [Model Surgery](/edgeai-modeloptimization/torchmodelopt/edgeai_torchmodelopt/xmodelopt/surgery/v2/README.md).

### Sparsity

    from edgeai_torchtoolkit import xmodelopt
    model = xmodelopt.pruning.PrunerModule(model, pruning_ratio=pruning_ratio, total_epochs=epochs, pruning_type=pruning_type)

> Here, desired pruning ratio, total training epochs, and the pruning type (Options : 'channel' (default), 'n2m', 'prunechannelunstructured', 'unstructured') needs to be specified.

This is the basic usage, the detailed usage for the API is documented in [Model Sparsity](/edgeai-modeloptimization/torchmodelopt/edgeai_torchmodelopt/xmodelopt/pruning/README.md).



# Results 
""" give overview of Different Trained Models and the obtained Accuracies as well as guidelines """


# FAQ


# Contributions



# License