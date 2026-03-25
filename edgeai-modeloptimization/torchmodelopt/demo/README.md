# Model Optimization Toolkit Demo
 A Demo for edgeai-modeloptimization tool kit developed using [Streamlit](https://streamlit.io/)
## Table of Content 
- [Model Optimization Toolkit Demo](#model-optimization-toolkit-demo)
  - [Table of Content](#table-of-content)
  - [Introduction](#introduction)
  - [How to run](#how-to-run)
  - [Limitations](#limitations)

## Introduction
This is a demo for edgeai-modeloptimization tool kit developed using **Streamlit**. This Demo uses the following repository to run:

- edgeai-torchvision
- edgeai-modeloptimization
- [imagenette](https://github.com/fastai/imagenette) : a subset of imagenet dataset with 10 class

## How to run
1. You can directly run the demo using the following
   1.  Streamlit
    ```sh
    streamlit run model_optimization_demo.py
    ```
   2. Shell script
    ```sh
    run_demo.sh
    ```

2. Once it is started, it  will give some IP addresses.
   
3. Open the IP address on a web browser to use the demo.

## Limitations 
- Right Now quantization is not enabled yet.
