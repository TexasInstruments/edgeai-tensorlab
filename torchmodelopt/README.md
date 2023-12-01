# EdgeAI Torch Model Optimization Toolkit

## Introduction
Tools to help development of Embedded Friendly Deep Neural Network Models in [Pytorch](https://pytorch.org) we call these **model optimization tools**. This contains [xmodelopt](xmodelopt) which is our extension of [torch.ao](https://github.com/pytorch/pytorch/tree/main/torch/ao) model architecture optimization tools. These are tools to make models more efficient and embedded friendly. <br>

## Overview
Torch model optimiation toolkit supports the following major features.

### Quantization:

- **Latest Quantization Aware Training / QAT (v2): [edgeai_torchmodelopt.xmodelopt.quantization.v2](./xmodelopt/quantization/v2)** - Easy to use wrapper over Pytorch native Quantization Aware Training (QAT) that uses **torch.fx** (built on top of **torch.ao** architecture optimization tools). Compared to the legacy QAT using **torch.nn** modules, torch.fx based QAT is able to handle models that uses torch operators and functionals as well. (**Note**: Models trained with this v2 QAT tool is supported in TIDL version 9.1  November 2023 onwards)<br>

- Legacy Quantization Aware Training / QAT (v1): [edgeai_torchmodelopt.xmodelopt.quantization.v1](./xmodelopt/quantization/v1) - Our legacy implementation of Quantization Aware Training (QAT) using **torch.nn** modules.<br>

### Model Surgery:

- **Latest Model surgery tool (v2)**: [edgeai_torchmodelopt.xmodelopt.surgery.v2](./xmodelopt/surgery/v2) - Easily replace layers (Modules, operators, functional) with other layers without modifying the model code. Easily replace unsupported layers (in a certain SoC) with other supported layers; create embedded friendly models. This uses torch.fx based surgery to handle models that uses torch modules, operators and functionals. (Compared to this, legacy surgery using **torch.nn** can only handle modules)<br>

- Legacy Model surgery tool (v1): [edgeai_torchmodelopt.xmodelopt.surgery.v1](./xmodelopt/surgery/v1) - Our legacy implementation of Model surgery using **torch.nn** modules.<br>

### Sparsity:

- Prune/sparsify a model: [edgeai_torchmodelopt.xmodelopt.pruning](./xmodelopt/pruning) - Both structured and unstructured pruning is supported. Structured pruning includes N:M pruning and channel pruning. Here, we provide with a few parametrization techniques, and the user can bring their own technique as well and implement as a part of the toolkit.
<br>

- Channel Sparsity uses torch.fx to find the connections and obtain a smaller network after the training with induced sparsity finishes. 

### Model Utilities:

- This package also contains edgeai_torchmodelopt.xnn which is our extension of torch.nn neural network model utilities. This is for advanced users and contains several undocumented utilities.

> ## Note
> The Quantization and Model Surgery tools are available in both (v1 and v2), but their interfaces and functionality are slightly different. We recommend to use the latest version (torch.fx based v2) tools whenever possible, but there may be situations in which legacy tools may be useful. For example the QAT models from torch.fx will be supported in TIDL only from the 9.1 release (November 2023). <br> <br>

<!-- 

### Supported Devices

-->

### Why use our toolkit?

Our toolkit provides the APIs for quantization, surgery as well as sparsity along with multiple torch.nn tools, for user to seemlessly introduce them in their own training code even for someone having basic knowledge of pytorch. 
The user can add a single line of code to introduce each of them as shown in the user guides. 
We have tested our algorithms which are the part of the toolkit and have obtained good results as shown in the results section.


# Getting Started

## Installation

### Package installation
Install the package for usage

    pip3 install "git+https://github.com/TexasInstruments/edgeai-modeloptimization.git#subdirectory=torchmodelopt"

### Source Installation
Install this repository as a local editable Python package (for development)

    cd edgeai-modeloptimization/torchmodelopt
    ./setup.sh


## User Guides

### Quantization

    import edgeai_torchmodelopt
    # wrap your model in xnn.quantization.QATFxModule. 
    # once it is wrapped, the actual model is in model.module
    model = edgeai_torchmodelopt.xmodelopt.quantization.v2.QATFxModule(model, total_epochs=epochs)

This is the basic usage, the detailed usage for the API is documented in [Quantization](/edgeai-modeloptimization/torchmodelopt/edgeai_torchmodelopt/xmodelopt/quantization/v2/README.md).

### Model Surgery

    # Using the default replacement dictionary in surgery
    import edgeai_torchmodelopt
    model = edgeai_torchmodelopt.xmodelopt.surgery.v2.convert_to_lite_fx(model)

<!-- # adding custom layers for replacement in surgery
import edgeai_torchmodelopt
replacement_dict = edgeai_torchmodelopt.xmodelopt.surgery.v2.get_replacement_dict_default()
replacement_dict.update({'layerNorm':custom_surgery_functions.replace_layer_norm})
model = edgeai_torchmodelopt.xmodelopt.surgery.v2.convert_to_lite_fx(model, replacement_dict=replacement_dict) -->

This is the basic usage, the detailed usage and adding the custom replacement dictionary for the API is documented in [Model Surgery](/edgeai-modeloptimization/torchmodelopt/edgeai_torchmodelopt/xmodelopt/surgery/v2/README.md).

### Sparsity

    from edgeai_torchmodelopt import xmodelopt
    model = xmodelopt.pruning.PrunerModule(model, pruning_ratio=pruning_ratio, total_epochs=epochs, pruning_type=pruning_type)

> Here, desired pruning ratio, total training epochs, and the pruning type (Options : 'channel', 'n2m', 'prunechannelunstructured', 'unstructured') needs to be specified.

This is the basic usage, the detailed usage for the API is documented in [Model Sparsity](/edgeai-modeloptimization/torchmodelopt/edgeai_torchmodelopt/xmodelopt/pruning/README.md).



# Results 

The results are using the torchvision models. The classification models are trained on the imagenet dataset and the detection models are trained using the coco dataset.

## Model Surgery

We use the default dictionary for model surgery. Here are the classification model results. 

| Models        | Torchvision Accuracy          | Lite Model Accuracy   |
| ------------- |:-------------:    | :-----:                |
| MobileNet_V3_Large_Weights.IMAGENET1K_V2  | 75.274    | 71.7                 |
| MobileNet_V2_Weights.IMAGENET1K_V2     | 72.154         |   72.88                 |
| ResNet18_Weights.IMAGENET1K_V1 | 69.758          |    69.758                 |
| ResNet50_Weights.IMAGENET1K_V2 | 80.9 | 80.9|
| ResNet101_Weights.IMAGENET1K_V2 | 81.9 | 81.9|
| ResNeXt50_32X4D_Weights.IMAGENET1K_V2 | 81.2 | 81.2 | 
| ResNeXt101_32X8D_Weights.IMAGENET1K_V2 | 82.83 | 82.83 |
| RegNet_X_400MF_Weights.IMAGENET1K_V2 | 74.86 | 74.86 | 
| RegNet_X_800MF_Weights.IMAGENET1K_V2 | 77.52 | 77.52 |
| RegNet_X_1_6GF_Weights.IMAGENET1K_V2 | 79.67 | 79.67 |

Here are the object detection model results.

| Models        |  Accuracy          | Lite Model Accuracy   |
| ------------- |:-------------:    | :-----:                |
| yolov5_nano  | 28.0    | 25.1                 |
| yolov5_small     | 37.7         |   35.5                |
| yolov7_tiny | 37.5          |    36.7                 |
| yolov8_nano | 37.2 | 34.5|
| yolov8_small | 44.2 | 42.4|


## Quantization

| Models        |  Accuracy          | Quantized Model Accuracy   |
| ------------- |:-------------:    | :-----:                |
| MobileNetv2  | 71.88 | 70.37           |
| ResNet50     | 76.13         |   76.02               |


## Model Sparsity

We here show results on pruning the network with n:m pruning and channel pruning using our blending based pruning algorithm.

Here are the results on 41:64 (n:m) pruning, that comes up to 0.640625 pruning ratio.

| Models        |  Accuracy          | Pruned Model Accuracy   |
| ------------- |:-------------:    | :-----:                |
| MobileNetv2  | 71.88 | 70.37           |
| ResNet50     | 76.13         |   76.02               |


Below are the results with networks having 30 \% channel sparsity. These networks could give upto 50% FLOP reduction and double the speedup. 

| Models        |  Accuracy          | Pruned Model Accuracy   |
| ------------- |:-------------:    | :-----:                |
| MobileNetv2  | 71.88 | 64.64          |
| ResNet50     | 76.13         |   74.07              |

# FAQ

Question 1: I am getting error "RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss." while training.

> Solution 1: Try setting 'find_unused_parameters=True' while wrapping the model in torch.nn.parallel.DistributedDataParallel. 

Question 2: Can I use different parts of the toolkit together?

> Solution 2: Surgery currently works with both sparsity and quantization individually, but sparsity together with quantization is cyrrently not supported. It will be made available soon. 


# Contributions

In case of any queries, you can directly contact [parakh08](https://github.com/parakh08) on p-agarwal@ti.com or [mathmanu](https://github.com/mathmanu) on mathew.manu@ti.com .
