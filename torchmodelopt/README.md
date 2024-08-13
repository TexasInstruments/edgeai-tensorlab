# EdgeAI Torch Model Optimization Toolkit

## Table of contents
1. [Introduction](#introduction)
2. [Installation] (#installation)
2. [Features](#features)
    1. [Quantization](#quantization)
    2. [Model Surgery](#model-surgery)
    3. [Sparsity](#sparsity)
    4. [Other Utilities](#other-utilities)
5. [FAQ](#faq)
6. [Contributors](#contributors) 


## Introduction
Tools to help development of Embedded Friendly Deep Neural Network Models in [Pytorch](https://pytorch.org). We call these **model optimization tools** because they help in making the models more efficient. This contains [xmodelopt](./edgeai_torchmodelopt/xmodelopt) which is our extension of [torch.ao](https://github.com/pytorch/pytorch/tree/main/torch/ao) model architecture optimization tools. <br>

## Note
The Quantization and Model Surgery tools are available in both (v1 and v2), but their interfaces and functionality are slightly different. We recommend to use the latest version (torch.fx based v2) tools whenever possible, but there are situations where the legacy tools may be useful - for example, for QAT models to be used with older TIDL versions or for older target devices such TDA4VM. <br> 

### Why use our toolkit?
- Our toolkit provides the APIs for quantization, surgery as well as pruning/sparsity along with other tools, for users to seamlessly introduce them in their own training code. 
- These are wrapped in easy to use interfaces so that even someone with basic knowledge of Pytorch can use these. The user can add a single line of code to introduce each of these functionality as shown in the user guides. 

## Getting Started

### Installation

#### Package installation
Install the package for usage

```
pip3 install "git+https://github.com/TexasInstruments/edgeai-modeloptimization.git#subdirectory=torchmodelopt"
```

#### Source Installation
Install this repository as a local editable Python package (for development)

```
cd edgeai-modeloptimization/torchmodelopt
./setup.sh
```

## Features
### Quantization:

[Quantization documentation](./edgeai_torchmodelopt/xmodelopt/quantization/README.md)


### Model Surgery:

[Model Surgery documentation](./edgeai_torchmodelopt/xmodelopt/surgery/README.md)


### Pruning/Sparsity:

[Model Pruning/Sparsity documentation](./edgeai_torchmodelopt/xmodelopt/pruning/README.md)


### Other Utilities:
- This package also contains edgeai_torchmodelopt.xnn which is our extension of torch.nn neural network model utilities. This is for advanced users and contains several undocumented features.


## FAQ

Question 1: I am getting error "RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss." while training.

> Solution 1: Try setting 'find_unused_parameters=True' while wrapping the model in torch.nn.parallel.DistributedDataParallel. 

Question 2: Can I use different parts of the toolkit together?

> Solution 2: Surgery currently works with both sparsity and quantization individually, but sparsity together with quantization is cyrrently not supported. It will be made available soon. 

Question 3: I am seeing a huge drop in accuracy while training when using multi-gpu configuration.

> Solution 3: Some configurations are seeing that, however, after the training, the model will be able to give the desired accuracy, otherwise single-gpu configuration could be used. 

Question 4: I am getting RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment.

> Solution 4: copy.deepcopy(model) is not supported while training and the training code might have it. It needs to be diabled and same model needs to be used. Make sure to keep the the model on same device though to enable training.

## Contributors

- @parakh08 [Parakh Agarwal]
- @mathmanu [Manu Mathew]
- @rakib23r [Rekib Zaman]
- @kpatel [Kunal Patel]
