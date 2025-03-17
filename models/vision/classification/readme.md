# Image Classification


## Introduction
Image classification is one of the most popular problems in Computer Vision. It has become a benchmark in Deep Learning for measuring the efficacy of neural architectures.

---

## Datasets
**ImageNet** Dataset: ImageNet Large Scale Visual Recognition Challenge (ILSVRC) (or just known as ImageNet) is a 1000 class subset of the whole ImageNet (http://image-net.org/) and this subset contains more than a million images in the training set and 50,000 images in the validation set. All the ImageNet accuracies in this page are measured on the validation set.

---

## Models 
The models are grouped in terms of repositories used to train them or the repositories through they are made available. Models from the following repositories are currently part of TI model zoo.

[EdgeAI-TorchVision](#EdgeAI-TorchVision)

[Torchvision](#Torchvision-models)

[EdgeAI-HF-Transformers](#EdgeAI-HF-Transformers) Transformer models repository forked from https://github.com/huggingface/transformers

[Facebookresearch/pycls](#Facebookresearch/pycls)

[Tensorflow-TPU](#Tensorflow-TPU-Models)

[TensorFlow-Model-Garden-TF1.0](#TensorFlow-Model-Garden-TF1.0)

[TensorFlow-Model-Garden-TF2.0-Keras](#TensorFlow-Model-Garden-TF2.0-Keras)

[MXNet](#MXNet-Models)

[github.com/onnx/models](#Models-From-github.com/onnx/models)

---

<div id="EdgeAI-TorchVision"></div>

### EdgeAI-TorchVision
EdgeAI-TorchVision is our extension of Torchvision that can be used to train embedded friendly Lite models.
- [Models Link](./imagenet1k/edgeai-tv/)
- **[Training source code](https://github.com/TexasInstruments/edgeai-torchvision)**
- [More Information](https://github.com/TexasInstruments/edgeai-torchvision/blob/master/docs/pixel2pixel/Image_Classification.md)

|Dataset  |Model Name             |Input Size|GigaMACs|Top-1 Accuracy%|Available|Notes    |
|---------|----------             |----------|--------|--------       |---------|------   |
|ImageNet |MobileNetV1[2]         |224x224   |0.568   |71.83          |Y        |         |
|ImageNet |MobileNetV2[3]         |224x224   |0.296   |72.13          |Y        |         |
|ImageNet |MobileNetV3Lite-Small[20]|224x224 |0.054   |62.688         |Y        |         |
|ImageNet |MobileNetV3Lite-Large[20]|224x224 |0.213   |72.122         |Y        |         |
|-
|ImageNet |MobileNetV2-QAT        |224x224   |0.296   |71.76          |Y        |QAT* Model|
|ImageNet |MobileNetV2-1.4-QAT    |224x224   |0.583   |74.57          |Y        |QAT* Model|
|ImageNet |MobileNetV3Lite-Small-QAT|224x224 |0.054   |61.836         |Y        |QAT* Model|
|ImageNet |MobileNetV3Lite-Large-QAT|224x224 |0.213   |71.614         |Y        |QAT* Model|

*- Quantization Aware Training using 8bit precision


---

<div id="Torchvision-models"></div>

### Torchvision models
Torchvision from Pytorch is one of the most popular packages for DNN training using PyTorch.
- [Models Link](./imagenet1k/torchvision/)
- [Additional information](https://pytorch.org/vision/stable/models.html)
- [Models Source Code](https://github.com/pytorch/vision)
- [**Training source code**](https://github.com/pytorch/examples/tree/master/imagenet)

|Dataset  |Model Name          |Input Size|GigaMACs|Top-1 Accuracy%|Available|Notes |
|---------|----------          |----------|--------|--------       |---------|------|
|ImageNet |MobileNetV2[3,5]    |224x224   |0.300   |72.00          |Y        |      |
|ImageNet |ShuffleNetV2[5,10]  |224x224   |0.151   |69.36          |Y        |      |
|ImageNet |ResNet18[4,10]      |224x224   |1.814   |69.76          |Y        |      |
|ImageNet |ResNet50[4,10]      |224x224   |4.087   |76.15          |Y        |      |
|ImageNet |VGG16[1,10]         |224x224   |15.35   |71.59          |         |      |
|ImageNet |RegNetX-400MF[6]    |224x224   |0.415   |72.83          |Y        |      |
|ImageNet |RegNetX-800MF[6]    |224x224   |0.800   |75.21          |Y        |      |
|ImageNet |RegNetX-1.6GF[6]    |224x224   |1.605   |77.04          |Y        |      |
|-
|ImageNet |MobileNetV2-QAT     |224x224   |0.300   |71.31          |Y        |QAT* Model|

*- Quantization Aware Training using 8b precision


---

<div id="EdgeAI-HF-Transformers"></div>

### Hugging Face Transformers
- [Models Link](./imagenet1k/hf-transformers/)
- [Training source code is in **edgeai-hf-transformers**](https://github.com/TexasInstruments/edgeai-tensorlab/edgeai-hf-transformers)
- Note: the above repository is forked and modified from: https://github.com/huggingface/transformers

|Dataset  | Model Name                               | Input Size | GigaMACs | Top-1 Accuracy% | Available |Notes    |
|---------|------------------------------------------|------------|----------|-----------------|-----------|------   |
|ImageNet | google/vit-base-patch16-224              | 224x224    |          | 75.404          |           |         |
|ImageNet | facebook/deit-tiny-patch16-224           | 224x224    |          | 72.132          |           |         |
|ImageNet | facebook/deit-small-patch16-224          | 224x224    |          | 79.9            |           |         |
|ImageNet | microsoft/swin-tiny-patch4-window7-224   | 224x224    | 4.5      | 81.2            |           |         |
|ImageNet | microsoft/swin-small-patch4-window7-224  | 224x224    | 8.7      | 83.2            |           |         |
|ImageNet | microsoft/swin-base-patch4-window7-224   | 224x224    | 15.4     | 84.81           |           |         |
|ImageNet | microsoft/swin-large-patch4-window7-224  | 224x224    |          | 86.146          |           |         |
|ImageNet | facebook/deit-tiny-patch16-224           | 224x224    |          | 72.2            |           |         |
|ImageNet | facebook/convnext-tiny-224               | 224x224    |          | 82.1            |           |         |
|ImageNet | facebook/convnext-small-224              | 224x224    |          | 83.1            |           |         |
|ImageNet | facebook/levit-128                       | 224x224    |          | 78.586          |           |         |
|ImageNet | facebook/levit-192                       | 224x224    |          | 79.874          |           |         |
|ImageNet | facebook/levit-256                       | 224x224    |          | 81.594          |           |         |
|ImageNet | facebook/levit-384                       | 224x224    |          | 82.594          |           |         |


---

<div id="Facebookresearch/pycls"></div>

### Facebookresearch/pycls
- [Models Link](./imagenet1k/fbr-pycls/)
- [Pretrained Models](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md)
- [Training source code](https://github.com/facebookresearch/pycls/)

|Dataset  |Model Name       |Input Size|GigaMACs|Top-1 Accuracy%|Available|Notes |
|---------|----------       |----------|--------|--------       |---------|------|
|ImageNet |RegNetX200MF[6]  |224x224   |0.200   |68.9           |Y        |      |
|ImageNet |RegNetX400MF[6]  |224x224   |0.400   |72.7           |Y        |      |
|ImageNet |RegNetX800MF[6]  |224x224   |0.800   |75.2           |Y        |      |
|ImageNet |RegNetX1.6GF[6]  |224x224   |1.600   |77.0           |Y        |      |

**Note**: Some of these models can also be trained using edgeai-torchvision


---

<div id="Tensorflow-TPU-Models"></div>

### Tensorflow TPU Models
- [Models Link](./imagenet1k/tf-tpu/)
- [Training source code](https://github.com/tensorflow/tpu/tree/master/models/official)
- [Training source code for EfficientNet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)

|Dataset  |Model Name                   |Input Size|GigaMACs|Top-1 Accuracy%|Available|Notes |
|---------|----------                   |----------|--------|--------       |---------|------|
|*Lite models*
|ImageNet |EfficientNet-Lite0[7,8]      |224x224   |0.407   |75.1           |Y        |      |
|ImageNet |EfficientNet-Lite1[7,8]      |240x240   |0.631   |76.7           |Y        |      |
|ImageNet |EfficientNet-Lite2[7,8]      |260x260   |0.899   |77.6           |         |      |
|ImageNet |EfficientNet-Lite4[7,8]      |300x300   |2.640   |80.2           |Y        |      |
|*Edge TPU models*
|ImageNet |EfficeintNet-EdgeTPU-S[7,9]  |224x224   |2.353   |77.23          |Y        |      |
|ImageNet |EfficeintNet-EdgeTPU-M[7,9]  |240x240   |3.661   |78.6           |Y        |      |
|ImageNet |EfficeintNet-EdgeTPU-L[7,9]  |300x300   |9.663   |80.2           |Y        |      |


---

<div id="TensorFlow-Model-Garden-TF1.0" ></div>

### TensorFlow Model Garden - Models Using Tensorflow 1.0 APIs
- [Models Link](./imagenet1k/tf1-models/)
- [Training source code](https://github.com/tensorflow/models)
- [More Information - Hosted Models](https://www.tensorflow.org/lite/models)
- [More Information - Slim Models](https://github.com/tensorflow/models/tree/master/research/slim)

|Dataset  | Model Name                         |Input Size| GigaMACs | Top-1 Accuracy% |Available| Notes |
|---------|------------------------------------|----------|----------|-----------------|---------|---|
|ImageNet | MobileNetV1[2,11]                  |224x224   | 0.569    | 71.0            |Y        |   |
|ImageNet | MobileNetV2[3,11]                  |224x224   | 0.301    | 71.9            |Y        |   |
|ImageNet | MobileNetV2-1.4[3,11]              |224x224   | 0.583    | 75.0            |Y        |   |
|ImageNet | SqueezeNet[12]                     |224x224   | 0.844    | 49.0            |Y        |   |
|ImageNet | DenseNet[13]                       |224x224   | 3.08     | 74.98           |Y        |   |
|ImageNet | InceptionV1[14,11]                 |224x224   | 1.512    | 69.63           |Y        |   |
|ImageNet | InceptionV3[15,11]                 |224x224   | 5.74     | 78.0            |Y        |   |
|ImageNet | MNasNet[16,11]                     |224x224   | 0.315    | 74.08           |Y        |   |
|ImageNet | NasNetMobile[17,11]                |224x224   | 0.136    | 73.9            |Y        |https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet   |
|ImageNet | MobileNetV3-Large-Minimalistic[20] |224x224   | 0.2099   | 72.3            |Y        |https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet   |
|ImageNet | MobileNetV3-Small-Minimalistic[20] |224x224   | 0.0526   | 61.9            |Y        |https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet   |



---

<div id="TensorFlow-Model-Garden-TF2.0-Keras" ></div>

### TensorFlow Model Garden - Models Using Tensorflow 2.0 Keras APIs
- [Models Link](./imagenet1k/tf2-models/)
- [Training source code](https://github.com/tensorflow/models)
- [More Information - Official Models](https://github.com/tensorflow/models/tree/master/official)
- [More Information - Keras Models](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/keras/applications)

|Dataset  |Model Name                   |Input Size|GigaMACs|Top-1 Accuracy%|Available|Notes |
|---------|----------                   |----------|--------|--------       |---------|------|
|ImageNet |ResNet50V1                   |224x224   |3.489   |75.2           |         |      |
|ImageNet |ResNet50V2                   |224x224   |3.498   |75.6           |         |      |



---

<div id="MXNet-Models" ></div>

### MxNet Models
- [Models Link](./imagenet1k/gluoncv-mxnet/)
- [More Information - GluonCV Model Zoo](https://cv.gluon.ai/model_zoo/index.html)

|Dataset  |Model Name                   |Input Size|GigaMACs|Top-1 Accuracy%|Available|Notes |
|---------|----------                   |----------|--------|--------       |---------|------|
|ImageNet |MobileNetV2                  |224x224   |0.314   |72.04          |Y        |      |
|ImageNet |ResNet50                     |224x224   |4.360   |79.15          |Y        |      |
|ImageNet |Xception                     |299x299   |13.939  |79.56          |Y        |      |



---

<div id="Models-From-github.com/onnx/models" ></div>

### Models From github.com/onnx/models
- [Models Link](./imagenet1k/onnx-models/)
- [More Information - ONNX Model Zoo](https://github.com/onnx/models)



---

## Notes
- GigaMACS: Complexity in Giga Multiply-Accumulations (lower is better). This is an important metric to watch out for when selecting models for embedded inference.<br>
- Input Size: Input resolution to the model. Note that input size often refers to the final crop resolution which may be different from the initial resize resolution. For example, the ImageNet mdels with Input Size of 224x224 are first resized such that the smaller side is 256 (=224/0.875) before cropping.<br>
- Top-1 Accuracy%: Floating Point Classification Accuracy (Top-1)% obtained in the validation set after training (accuracy reported by the training script / training repository).<br>
- PyTorch models are exported to ONNX format using torch.onnx.export()
- Tensorflow models are exported to TFLite format as explained in the TFLite documentation.

---

## References

[1] Very Deep Convolutional Networks for Large-Scale Image Recognition, K. Simonyan, A. Zisserman, International Conference on Learning Representations, 2015

[2] MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, Howard AG, Zhu M, Chen B, Kalenichenko D, Wang W, Weyand T, Andreetto M, Adam H, arXiv:1704.04861, 2017

[3] MobileNetV2: Inverted Residuals and Linear Bottlenecks, Sandler M, Howard A, Zhu M, Zhmoginov A, Chen LC. arXiv preprint. arXiv:1801.04381, 2018.

[4] Deep Residual Learning for Image Recognition, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, https://arxiv.org/abs/1512.03385

[5] PyTorch TorchVision Model Zoo: https://pytorch.org/docs/stable/torchvision/models.html

[6] Designing Network Design Spaces, Ilija Radosavovic Raj Prateek Kosaraju Ross Girshick Kaiming He Piotr Dollar´, Facebook AI Research (FAIR), https://arxiv.org/pdf/2003.13678.pdf, https://github.com/facebookresearch/pycls

[7] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, Mingxing Tan, Quoc V. Le, https://arxiv.org/abs/1905.11946

[8] EfficientNet-lite are a set of mobile/IoT friendly image classification models. https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/README.md

[9] EfficientNet-EdgeTPU are a family of image classification neural network models customized for deployment on Google Edge TPU. https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu

[10] ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design, Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun, https://arxiv.org/abs/1807.11164

[11] TensorFlow Model Garden: The TensorFlow Model Garden is a repository with a number of different implementations of state-of-the-art (SOTA) models and modeling solutions for TensorFlow users. https://github.com/tensorflow/models

[12] SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size, Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer, https://arxiv.org/abs/1602.07360

[13] Densely Connected Convolutional Networks, Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger, https://arxiv.org/abs/1608.06993

[14] Going Deeper with Convolutions, Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich, https://arxiv.org/abs/1409.4842

[15] Rethinking the Inception Architecture for Computer Vision, Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, https://arxiv.org/abs/1512.00567

[16] MnasNet: Platform-Aware Neural Architecture Search for Mobile, Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, Quoc V. Le, https://arxiv.org/abs/1807.11626

[17] Learning Transferable Architectures for Scalable Image Recognition, Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V. Le, https://arxiv.org/abs/1707.07012

[18] ONNX Model Zoo: The ONNX Model Zoo is a collection of pre-trained, state-of-the-art models in the ONNX format contributed by community members like you. https://github.com/onnx/models

[19] GluonCV Model Zoo: GluonCV provides implementations of state-of-the-art (SOTA) deep learning algorithms in computer vision. https://cv.gluon.ai/model_zoo/index.html

[20] Searching for MobileNetV3, Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam, https://arxiv.org/abs/1905.02244

