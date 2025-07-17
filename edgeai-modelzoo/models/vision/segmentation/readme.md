# Semantic Segmentation  Benchmark


## Introduction
Semantic Segmentation from Image (or Image Segmentation) is an important problem in Computer Vision. This page describes some of the popular Semantic Segmentation models.


## Datasets

COCO dataset: Microsoft COCO: Common Objects in Context, Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár, https://arxiv.org/abs/1405.0312, https://cocodataset.org 

**COCOSeg21** dataset: For information about 21 class subset of COCO Segmentation, see the description in [torchvision models](https://pytorch.org/vision/stable/models.html) 

**Cityscapes** dataset: Cityscapes is a large scale dataset containing street scenes from 50 different cities with high quality pixel level annotations.

**ADE20K** dataset: Scene Parsing Dataset Scene Parsing through ADE20K Dataset. Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso and Antonio Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. Semantic Understanding of Scenes through ADE20K Dataset. Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso and Antonio Torralba. International Journal on Computer Vision (IJCV). https://groups.csail.mit.edu/vision/datasets/ADE20K/, http://sceneparsing.csail.mit.edu/

**ADE20K32:** 32 class dataset subset of ADE20K described in the paper: MLPerf Inference Benchmark, Vijay Janapa Reddi et. al., https://arxiv.org/abs/1911.02549


## Models

Note: **Transformer models have been added using the edgeai-hf-transformers repository (see below)**


### edgeai-tensorvision
- [Models Link - COCOSeg21](./cocoseg21/edgeai-tv/)
- [Models Link - ADE20K32](./ade20k32/edgeai-tv/)
- [Additional information](https://github.com/TexasInstruments/edgeai-tensorlab/edgeai-tensorvision/blob/master/docs/pixel2pixel/Semantic_Segmentation.md)
- [Training Code](https://github.com/TexasInstruments/edgeai-tensorlab/edgeai-tensorvision)


|Dataset    |Model Name                     |Input Size |GigaMACs  |MeanIoU%       |Available|Notes |
|-----------|------------------------------ |-----------|----------|---------------|---------|------|
|           |**ADE20K32 dataset models**
|ADE20K32   |MobileNetV2S16+DeepLabV3Lite   |512x512    |3.28      |51.01          |Y        |      | 
|ADE20K32   |MobileNetV2+UNetLite           |512x512    |2.427     |49.95          |Y        |      |
|ADE20K32   |MobileNetV2+FPNLite            |512x512    |3.481     |50.72          |Y        |      |
|ADE20K32   |MobileNetV2-1.4+FPNLite        |512x512    |6.646     |52.93          |Y        |      |
|- 
|ADE20K32   |RegNetX400MF+FPNLite           |384x384    |3.1526    |51.03          |Y        |      |
|ADE20K32   |RegNetX800MF+FPNLite           |512x512    |8.0683    |53.29          |Y        |      |
|           |**COCOSeg21 dataset models**
|COCOSeg21  |MobileNetV2S16+DeepLabV3Lite   |512x512    |3.161     |57.77          |Y        |      | 
|COCOSeg21  |MobileNetV2+UNetLite           |512x512    |2.009     |57.01          |         |      | 
|COCOSeg21  |MobileNetV2+FPNLite            |512x512    |3.357     |               |         |      | 
|COCOSeg21  |RegNetX800MF+FPNLite           |512x512    |7.864     |61.15          |Y        |      | 
|           |**Cityscapes dataset models**
|Cityscapes |MobileNetV2S16+DeepLabV3Lite   |768x384    |**3.54**  |**69.13**      |         |      |
|Cityscapes |MobileNetV2+UNetLite           |768x384    |**2.20**  |**68.94**      |         |      |
|Cityscapes |MobileNetV2+FPNLite            |768x384    |**3.84**  |**70.39**      |         |      |
|Cityscapes |MobileNetV2+FPNLite            |1536x768   |**15.07** |**74.61**      |         |      |
|-
|Cityscapes |MobileNetV2S16+DeepLabV3Lite-QAT |768x384  |**3.54**  |**68.77**      |         |QAT* model |
|Cityscapes |MobileNetV2+UNetLite-QAT         |768x384  |**2.20**  |**68.18**      |         |QAT* model |
|Cityscapes |MobileNetV2+FPNLite-QAT          |768x384  |**3.84**  |**69.88**      |         |QAT* model |
|-
|Cityscapes |RegNetX800MF+FPNLite           |768x384    |**8.84**  |**72.01**      |         |      |
|Cityscapes |RegNetX1.6GF+FPNLite           |1024x512   |**24.29** |**75.84**      |         |      |
|Cityscapes |RegNetX3.2FF+FPNLite           |1536x768   |**111.16**|**78.90**      |         |      |
|Cityscapes |RegNetX400MF+FPNLite           |768x384    |**6.09**  |**68.03**      |         |      |
|Cityscapes |RegNetX400MF+FPNLite           |1536x768   |**24.37** |**73.96**      |         |      |

*- Quantization Aware Training using 8b precision


### edgeai-torchvision / torchvision Models
- [Models Link - COCOSeg21](./cocoseg21/edgeai-tv/)
- [Models Link - ADE20K32](./ade20k32/edgeai-tv/)
- [Training Code](https://github.com/TexasInstruments/edgeai-tensorlab/edgeai-torchvision)
- [Original Training source code](https://github.com/pytorch/vision/tree/master/references)
- [Additional information](https://pytorch.org/vision/stable/models.html)

|Dataset    |Model Name                     |Input Size |GigaMACs  |MeanIoU%       |Available|Notes |
|---------- |-------------------------------|-----------|----------|---------------|---------|------|
|           |**COCOSeg21 dataset models**
|COCOSeg21  |LR-ASPP MobileNetV3-Large[11]  |520        |          |57.9           |         |      |
|COCOSeg21  |DeepLabV3 MobileNetV3-Large[11]|520        |          |60.3           |         |      |
|COCOSeg21  |DeepLabV3 ResNet50[11]         |520        |          |66.4           |         |      |
|           |**Cityscapes dataset models**
|Cityscapes |ResNet50+FCN[3,11]             |1040x520   |285.4     |71.6           |         |      |
|Cityscapes |ResNet50+DeepLabV3[5,11]       |1040x520   |337.5     |73.5           |         |      |


### edgeai-hf-transformers
- [Models Link](./coco/hf-transformers/)
- **[Training source code](https://github.com/TexasInstruments/edgeai-tensorlab/edgeai-hf-transformers)**
- Note: the above repository is forked and modified from: https://github.com/huggingface/transformers

|Dataset | Model Name                                 | Input Size | GigaMACS | MeanIoU% | Available |Notes |
|--------|--------------------------------------------|------------|----------|----------|-----------|----- |
|ADE20K  | nvidia/segformer-b0-finetuned-ade-512-512  | 512x512    | 8.4      | 37.4     |           |      |  
|ADE20K  | nvidia/segformer-b5-finetuned-ade-640-640  | 640x640    | 95.7     | 51.1     |           |      |  


### Tensorflow DeepLab Models
- [Models Link](./voc2012/tf1-models/)
- [Training source code](https://github.com/tensorflow/models/tree/master/research/deeplab)
- [**Additional Information**](./utils/tf-deeplab)

|Dataset    |Model Name                     |Input Size |GigaMACs  |MeanIoU%    |Available|Notes |
|---------- |------------------------------ |-----------|----------|------------|---------|------|
|-          |**Our modified models**
|VOC2012    |deeplabv3_mnv2_dm05[10]        |512x512    |2.77      |66.94       |Y        |      |
|VOC2012    |deeplabv3_mnv2[10]             |512x512    |8.60      |72.66       |Y        |      |
|VOC2012    |deeplabv3_xception[10]         |512x512    |171.77    |81.74       |         |      |
|-          |**Original DeepLab models**
|Cityscapes |MobileNetV2+DeepLab[10]        |769x769    |21.27     |70.71       |         |      |
|Cityscapes |MobileNetV3+DeepLab[10]        |769x769    |15.95     |72.41       |         |      |
|Cityscapes |Xception65+DeepLab[10]         |769x769    |418.64    |78.79       |         |      |


### Other Miscellaneous Models
These are for reference only and are not included in this Model Zoo.

|Dataset    |Model Name                    |Input Size  |GigaMACs  |MeanIoU%       |Available|Notes |
|-----------|------------------------------|------------|----------|---------------|---------|------|
|Cityscapes |ERFNet[8]                     |1024x512    |27.705    |69.7           |         |      |
|Cityscapes |MobileNetV2+SwiftNet[9]       |2048x1024   |41.0      |75.3           |         |      |


## Notes
- The suffix 'Lite' in the name of models such as DeepLabV3Lite, FPNLite & UNetLite indicates the use of Depthwise convolutions or Grouped convolutions. If the feature extractor (encoder) uses Depthwise Convolutions, then Depthwise convolutions are used throughout such models - even in the neck and decoder. If the feature extractor (encoder) uses grouped convolutions as in the case of RegNetX, then grouped convolutions (with the same group size as that of the feature extractor) are used even in the neck and decoder.<br>
- GigaMACS: Complexity in Giga Multiply-Accumulations (lower is better). This is an important metric to watch out for when selecting models for embedded inference.<br>
- MeanIoU%: Original Floating Point MeanIoU Accuracy% obtained after training.<br>


## References

[1] The PASCAL Visual Object Classes (VOC) Challenge, Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman, A.
International Journal of Computer Vision, 88(2), 303-338, 2010, http://host.robots.ox.ac.uk/pascal/VOC/

[2] The Cityscapes Dataset for Semantic Urban Scene Understanding, Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, Bernt Schiele, https://arxiv.org/abs/1604.01685

[3] Fully Convolutional Networks for Semantic Segmentation, Jonathan Long, Evan Shelhamer, Trevor Darrell, https://arxiv.org/abs/1411.4038

[4] Feature Pyramid Networks for Object Detection Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie, https://arxiv.org/abs/1612.03144

[5] Rethinking Atrous Convolution for Semantic Image Segmentation, Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam, https://arxiv.org/abs/1706.05587

[6] U-Net: Convolutional Networks for Biomedical Image Segmentation, Olaf Ronneberger, Philipp Fischer, Thomas Brox, https://arxiv.org/abs/1505.04597

[7] Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam, https://arxiv.org/abs/1802.02611

[8] E. Romera, J. M. Alvarez, L. M. Bergasa, and R. Arroyo. Erfnet: Efficient residual factorized convnet for real-time semantic segmentation. IEEE Transactions
on Intelligent Transportation Systems, 19(1):263272, 2018.

[9] In Defense of Pre-trained ImageNet Architectures for Real-time Semantic Segmentation of Road-driving Images, Marin Oršić*, Ivan Krešo*, Siniša Šegvić, Petra Bevandić (* denotes equal contribution), CVPR, 2019.

[10] Tensorflow DeepLab ModelZoo, https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md

[11] Torchvision Models, https://pytorch.org/docs/stable/torchvision/models.html, https://github.com/pytorch/vision/tree/master/torchvision

