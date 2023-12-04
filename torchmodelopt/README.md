# EdgeAI Torch Model Optimization Toolkit

## Introduction
Tools to help development of Embedded Friendly Deep Neural Network Models in [Pytorch](https://pytorch.org) we call these **model optimization tools**. This contains [xmodelopt](./edgeai_torchmodelopt/xmodelopt) which is our extension of [torch.ao](https://github.com/pytorch/pytorch/tree/main/torch/ao) model architecture optimization tools. These are tools to make models more efficient and embedded friendly. <br>

## Overview
Torch model optimiation toolkit supports the following major features.

### Quantization:

- **Latest Quantization Aware Training / QAT (v2): [edgeai_torchmodelopt.xmodelopt.quantization.v2](./edgeai_torchmodelopt/xmodelopt/quantization/v2)** - Easy to use wrapper over Pytorch native Quantization Aware Training (QAT) that uses **torch.fx** (built on top of **torch.ao** architecture optimization tools). Compared to the legacy QAT using **torch.nn** modules, torch.fx based QAT is able to handle models that uses torch operators and functionals as well. (**Note**: Models trained with this v2 QAT tool is supported in TIDL version 9.1  November 2023 onwards)<br>

- Legacy Quantization Aware Training / QAT (v1): [edgeai_torchmodelopt.xmodelopt.quantization.v1](./edgeai_torchmodelopt/xmodelopt/quantization/v1) - Our legacy implementation of Quantization Aware Training (QAT) using **torch.nn** modules.<br>

### Model Surgery:

- **Latest Model surgery tool (v2)**: [edgeai_torchmodelopt.xmodelopt.surgery.v2](./edgeai_torchmodelopt/xmodelopt/surgery/v2) - Easily replace layers (Modules, operators, functional) with other layers without modifying the model code. Easily replace unsupported layers (in a certain SoC) with other supported layers; create embedded friendly models. This uses torch.fx based surgery to handle models that uses torch modules, operators and functionals. (Compared to this, legacy surgery using **torch.nn** can only handle modules)<br>

- Legacy Model surgery tool (v1): [edgeai_torchmodelopt.xmodelopt.surgery.v1](./edgeai_torchmodelopt/xmodelopt/surgery/v1) - Our legacy implementation of Model surgery using **torch.nn** modules.<br>

### Sparsity:

- Prune/sparsify a model: [edgeai_torchmodelopt.xmodelopt.pruning](./edgeai_torchmodelopt/xmodelopt/pruning) - Both structured and unstructured pruning is supported. Structured pruning includes N:M pruning and channel pruning. Here, we provide with a few parametrization techniques, and the user can bring their own technique as well and implement as a part of the toolkit.
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

The details of the quantization is available at [quantization landing page](./edgeai_torchmodelopt/xmodelopt/quantization/v2/README.md) and user guide for the QAT API is documented in [QAT](./edgeai_torchmodelopt/xmodelopt/quantization/v2/docs/qat.md).

### Model Surgery

    # Using the default replacement dictionary in surgery
    import edgeai_torchmodelopt
    model = edgeai_torchmodelopt.xmodelopt.surgery.v2.convert_to_lite_fx(model)

<!-- # adding custom layers for replacement in surgery
import edgeai_torchmodelopt
replacement_dict = edgeai_torchmodelopt.xmodelopt.surgery.v2.get_replacement_dict_default()
replacement_dict.update({'layerNorm':custom_surgery_functions.replace_layer_norm})
model = edgeai_torchmodelopt.xmodelopt.surgery.v2.convert_to_lite_fx(model, replacement_dict=replacement_dict) -->

This is the basic usage, the detailed usage and adding the custom replacement dictionary for the API is documented in [Model Surgery](./edgeai_torchmodelopt/xmodelopt/surgery/v2/README.md).

### Sparsity

    from edgeai_torchmodelopt import xmodelopt
    model = xmodelopt.pruning.PrunerModule(model, pruning_ratio=pruning_ratio, total_epochs=epochs, pruning_type=pruning_type)

> Here, desired pruning ratio, total training epochs, and the pruning type (Options : 'channel', 'n2m', 'prunechannelunstructured', 'unstructured') needs to be specified.

This is the basic usage, the detailed usage for the API is documented in [Model Sparsity](./edgeai_torchmodelopt/xmodelopt/pruning/README.md).



# Results 

The results are using the torchvision models. The classification models are trained on the imagenet dataset and the detection models are trained using the coco dataset.

## Model Surgery

We use the default dictionary for model surgery. Here are the classification model results. The models that we obtain after the model surgery are called as lite models.


| Models        | Torchvision Accuracy          | Lite Model Accuracy   |
| ------------- |:-------------:    | :-----:                |
| MobileNet_V2   | 72.154         |   72.88                 |
| MobileNet_V3_Large  | 75.274    | 71.7                 |
| EfficientNet_B0 | 77.69 | 73.57* |
| EfficientNet_B1 | 79.83 | 74.49* |

Imagenet1k dataset has been used to train these models. We used the torchvision results from the [official website](https://pytorch.org/vision/stable/models.html).

\* The lite modes are just trained for 150 epochs against the suggested 600 epochs from torchvision training recipe. 

Here are the object detection model results trained on the COCO dataset using [mmyolo](https://github.com/open-mmlab/mmyolo) package. The training recipe were also adopted from the same package.

| Models        |  Accuracy          | Lite Model Accuracy   |
| ------------- |:-------------:    | :-----:                |
| yolov5_nano  | 28.0    | 25.1                 |
| yolov5_small     | 37.7         |   35.5                |
| yolov7_tiny | 37.5          |    36.7                 |
| yolov8_nano | 37.2 | 34.5|
| yolov8_small | 44.2 | 42.4|


## Quantization

Following are the results of 8 bit quantization of torchvision models and their lite alternatives. There is a marginal drop in accuracy for these networks. The networks are trained on imagenet dataset using the torchvision training package.

| Models        |  Float Accuracy          | Int8 Quantized Model Accuracy   |
| ------------- |:-------------:    | :-----:                |
| MobileNetv2-lite  | 72.938 | 72.476           |
| ResNet50     | 76.13         |   75.052              |


## Model Sparsity

Here are the results on pruning the network with n:m pruning and channel pruning using our blending based pruning algorithm.

Below are the results with networks having 30% channel sparsity. These networks could give upto 50% FLOP reduction and double the speedup. 

| Models        |  Accuracy          | Pruned Model Accuracy   |
| ------------- |:-------------:    | :-----:                |
| ResNet50     | 76.13         |   74.07              |

Here are the results on 41:64 (n:m) pruning, that comes up to 0.640625 pruning ratio.

| Models        |  Accuracy          | Pruned Model Accuracy   |
| ------------- |:-------------:    | :-----:                |
| MobileNetv2  | 71.88 | 70.37           |
| ResNet50     | 76.13         |   76.02               |



> <h3> The results obtained are preliminary results, and have scope for further optimizations. </h3>

# FAQ

Question 1: I am getting error "RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss." while training.

> Solution 1: Try setting 'find_unused_parameters=True' while wrapping the model in torch.nn.parallel.DistributedDataParallel. 

Question 2: Can I use different parts of the toolkit together?

> Solution 2: Surgery currently works with both sparsity and quantization individually, but sparsity together with quantization is cyrrently not supported. It will be made available soon. 


# Contributors

- @parakh08 [Parakh Agarwal]
- @mathmanu [Manu Mathew]
- @rakib23r [Rekib Zaman]