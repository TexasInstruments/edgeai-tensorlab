# Quantization

Quantization of a CNN model is the process of converting floating point data & operations to fixed point (integer). This includes quantization of weights, feature maps and all operations (including convolution). The quantization style used in this code is **Power-Of-2, Symmetric, Per-Tensor Quantization** for both **Weights and Activations**.

Accuracy of inference can degrade if the CNN model is quantized to 8bits using simple methods and steps have to be taken to minimize this accuracy loss. The parameters of the model need to be adjusted to suit quantization. This includes adjusting of weights, biases and activation ranges. This adjustment can be done as part of Post Training Quantization (PTQ) or as part of Quantization Aware Training (QAT).

## Overview
Inference engines use fixed point arithmetic to implement neural networks efficiently. For example TI Deep Learning Library (TIDL) for TI’s Jacinto7 TDA4x Devices (eg. TDA4VM, TDA4AL, TDA4VL etc) or AM6XA Devices (eg. AM62A, AM68A, AM69A etc) supports 16-bit and 8-bit fixed point inference modes.

Fixed point mode, especially the 8-bit mode can have accuracy degradation. The tools and guidelines provided here help to avoid accuracy degradation with quantization.

If you are getting accuracy degradation with 8-bit inference, the first thing to check is 16-bit inference. If 16-bit inference provides accuracy close to floating point and 8-bit has an accuracy degradation, there it is likely that the degradation si due to quantization. However, if there is substantial accuracy degradation with 16-bit inference itself, then it is likely that there is some issue other than quantization.  


### Quantization Schemes
Post Training Quantization (PTQ): Post Training Quantization involves range estimation for weights and activations and also minor tweaks to the model (such as bias adjustments). TIDL can accept a floating point model and do PTQ using a few sample images. This is done during the import of the model in TIDL. More details are in [**Post Training Quantization in TIDL (PTQ) documentation**](./docs/tidl_ptq.md).<br>

Quantization Aware Training (QAT): This is needed only if the accuracy obtained with PTQ is not satisfactory. QAT operates as a second phase after the initial training in floating point, in the training framework. More details are in [**Quantization Aware Training (QAT) documentation**](./docs/qat.md).

In this repository, we have  guidelines on how to choose models and how train them to get the best accuracy with Quantization. It is unlikely that there will be significant accuracy drop with PTQ if these guidelines are followed. In spite of this, if there are models that have significant accuracy drop with quantization, it is possible to improve the accuracy using QAT. Please read more details in the documentation for QAT.

## Results

The table below shows accuracy of several models obtained using QAT and comparison with PTQ models. 

| Task                  | Dataset    | Model Name                   | Input Size | GigaMACs | Accuracy(Float)% PyTorch | Accuracy(Int8)% TIDL-PTQ  | Accuracy(Int8)% PyTorch-QAT | Accuracy(Int8)% QAT Model in TIDL |
|-----------------------|------------|------------------------------|------------|----------|--------------------------|---------------------------|-----------------------------|-----------------------------------|
| Image Classification  | ImageNet   | MobileNetV1                  | 224x224    | 0.568    | 71.83                    | 70.512                    |                             |                                   |
| Image Classification  | ImageNet   | MobileNetV2                  | 224x224    | 0.296    | 72.13                    | 71.062                    | 71.76                       | 71.706                            |
| Image Classification  | ImageNet   | MobilenetV2(TV)              | 224x224    | 0.300    | 72.00                    | 67.642                    | 71.31                       | 71.116                            |
| Image Classification  | ImageNet   | MobileNetV3Lite-Small        | 224x224    | 0.054    | 62.688                   | 58.462                    | 61.836                      | 61.578                            |
| Image Classification  | ImageNet   | MobileNetV3Lite-Large        | 224x224    | 0.213    | 72.122                   | 71.04                     | 71.614                      |                                   |
| -
| Semantic Segmentation | Cityscapes | MobileNetV2S16+DeepLabV3Lite | 768x384    | 3.54     | 69.13                    | 66.83                     | 68.77                       |                                   |
| Semantic Segmentation | Cityscapes | MobileNetV2+UNetLite         | 768x384    | 2.20     | 68.94                    | 66.06                     | 68.18                       |                                   |
| Semantic Segmentation | Cityscapes | MobileNetV2+FPNLite          | 768x384    | 3.84     | 70.39                    | 67.24                     | 69.88                       |                                   |
| Semantic Segmentation | Cityscapes | RegNetX800MF+FPNLite         | 768x384    | 8.90     | 72.01                    | 71.81                     |                             |                                   |
| Semantic Segmentation | Cityscapes | RegNetX1.6GF+FPNLite         | 1024x512   | 26.49    | 75.84                    | 75.45                     |                             |                                   |
| Semantic Segmentation | Cityscapes | RegNetX3.2GF+FPNLite         | 1536x768   | 111.46   | 78.90                    | 78.80                     |                             |                                   |


Notes:<br>
- For Image Classification, the accuracy measure used is % Top-1 Classification Accuracy. <br>
- For Semantic Segmentation, the accuracy measure used in % MeanIoU Accuracy.
- The PTQ and QAT results use Power-Of-2, Symmetric, Per-Tensor, 8-bit Quantization.<br>
- PTQ in TIDL produces reasonable accuracy and should be sufficient for most models. But QAT is able to reduce the Accuracy gap even further.<br>
- (TV) indicates that the model is from torchvision Model Zoo<br>
- More details of these models can be seen in [edgeai-modelzoo](https://github.com/TexasInstruments/edgeai-modelzoo)<br>
- The TIDL results were obtained using [edgeai-benchmark](https://github.com/TexasInstruments/edgeai-benchmark) which in turn uses [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) and the release version used was 8.2<br>


## Post Training Calibration For Quantization (Calibration)
**Note: this is not our recommended method in PyTorch.**<br>

We also have a faster, but less accurate alternative called Calibration. Post Training Calibration or simply Calibration is a method to reduce the accuracy loss with quantization. This is an approximate method and does not use ground truth or back-propagation. Details are in the [documentation of Calibration](./docs/calibration.md). However, in a training framework such as PyTorch, it is possible to get better accuracy with QAT and we recommend to use that.<br>


## References 
[1] PACT: Parameterized Clipping Activation for Quantized Neural Networks, Jungwook Choi, Zhuo Wang, Swagath Venkataramani, Pierce I-Jen Chuang, Vijayalakshmi Srinivasan, Kailash Gopalakrishnan, arXiv preprint, arXiv:1805.06085, 2018

[2] Estimating or propagating gradients through stochastic neurons for conditional computation. Y. Bengio, N. Léonard, and A. Courville. arXiv preprint arXiv:1308.3432, 2013.

[3] Understanding Straight-Through Estimator in training activation quantized neural nets, Penghang Yin, Jiancheng Lyu, Shuai Zhang, Stanley Osher, Yingyong Qi, Jack Xin, ICLR 2019

[4] Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference, Benoit Jacob Skirmantas Kligys Bo Chen Menglong Zhu, Matthew Tang Andrew Howard Hartwig Adam Dmitry Kalenichenko, arXiv preprint, arXiv:1712.05877

[5] Trained quantization thresholds for accurate and efficient fixed-point inference of Deep Learning Neural Networks, Sambhav R. Jain, Albert Gural, Michael Wu, Chris H. Dick, arXiv preprint, arXiv:1903.08066 

[6] Quantizing deep convolutional networks for efficient inference: A whitepaper, Raghuraman Krishnamoorthi, arXiv preprint, arXiv:1806.08342

[7] TensorFlow / Learn / For Mobile & IoT / Guide / Post-training quantization, https://www.tensorflow.org/lite/performance/post_training_quantization

[8] QUANTIZATION / Introduction to Quantization, https://pytorch.org/docs/stable/quantization.html

[9] Designing Network Design Spaces, Ilija Radosavovic Raj Prateek Kosaraju Ross Girshick Kaiming He Piotr Dollar´, Facebook AI Research (FAIR), https://arxiv.org/pdf/2003.13678.pdf, https://github.com/facebookresearch/pycls

