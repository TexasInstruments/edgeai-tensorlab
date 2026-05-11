# Semantic Segmentation
Semantic segmentation assigns a class to each pixel of the image. It is useful for tasks such as lane detection, road segmentation etc. 

Commonly used Training/Validation commands are listed in the file [run_edgeailite_segmentation.sh](../../run_edgeailite_segmentation.sh). Uncommend one line and run the file to start the run. 

## Model Configurations
A set of common example model configurations are defined for all pixel to pixel tasks. These models can support multiple inputs (for example image and optical flow) as well as support multiple decoders for multi-task prediction (for example semantic segmentation + depth estimation + motion segmentation). 

Whether to use multiple inputs or how many decoders to use are fully configurable. This framework is also flexible to add different model architectures or backbone networks. Some of the model configurations are currently available are:

These are some of the supported models that use MobileNetV2 backbone.

**deeplabv3plus_mobilenetv2_tv_edgeailite**: (default) This model is mostly similar to the DeepLabV3+ model [[7]] using MobileNetV2 backbone. The difference with DeepLabV3+ is that we removed the convolutions after the shortcut and kep one set of depthwise separable convolutions to generate the prediction. The ASPP module that we used is a lite-weight variant with depthwise separable convolutions (DWASPP). We found that this reduces complexity without sacrificing accuracy. Due to this we call this model DeepLabV3+(Lite) or simply  DeepLabV3PlusEdgeAILite. (Note: The suffix "_tv" is used to indicate that our backbone model is from torchvision)

**unet_aspp_mobilenetv2_tv_edgeailite**: UNet [6] based edgeailite model.

**fpn_aspp_mobilenetv2_tv_edgeailite**: This is similar to Feature Pyramid Network [[4]], but adapted for edgeailite tasks. We stop the decoder at a stride of 4 and then upsample to the final resolution from there. We also use DWASPP module to improve the receptive field. We call this model FPNPixel2Pixel.

**fpn_aspp_mobilenetv2_tv_fd_edgeailite**: This is also FPN, but with a larger encoder stride(64). This is a low complexity model (using Fast Downsampling Strategy [12]) that can be used with higher resolutions.


**We find that RegNetX models strike a good balance between accuracy, model complexity, speed of inference on device and easiness of quantization.**  For RegNetX based edgeailite models, the same group size used in the encoder is used in the decoder part as well. Following are some of the RegNetX based models that are supported.

**deeplabv3plus_regnetx800mf_edgeailite**: RegNetX-800MF based DeepLabV3Plus model.

**unet_aspp_regnetx800mf_edgeailite**: RegNetX-800MF based UNet model.

**fpn_aspp_regnetx400mf_edgeailite**: RegNetX-400MF based FPN model.

**fpn_aspp_regnetx800mf_edgeailite** RegNetX-800MF based FPN model.

**fpn_aspp_regnetx1p6gf_edgeailite** RegNetX-1.6GF based FPN model.

**fpn_aspp_regnetx3p2gf_edgeailite** RegNetX-3.2GF based FPN model.


## Datasets: Cityscapes Dataset 
Download the Cityscapes dataset [[2]] from https://www.cityscapes-dataset.com/. You will need need to register before the data can be downloaded. Unzip the data into the folder ./data/datasets/cityscapes/data. This folder should contain leftImg8bit and gtFine folders of cityscapes. 

## Datasets: VOC Dataset 
The PASCAL VOC dataset [[1]] can be downloaded using the following:<br>
```
mkdir ./data/datasets/voc
cd /data/datasets/voc
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```

Extact the dataset files into ./data/datasets/voc/VOCdevkit
```
tar -xvf VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
```

Download Extra annotations: Download the augumented annotations as explained here: https://github.com/DrSleep/tensorflow-deeplab-resnet. For this, using a browser, download the zip file SegmentationClassAug.zip from: https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0

Unzip SegmentationClassAug.zip and place the images in the folder ./data/datasets/voc/VOCdevkit/VOC2012/SegmentationClassAug

Create a list of those images in the ImageSets folder using the following:<br>
```
cd VOCdevkit/VOC2012/ImageSets/Segmentation
ls -1 SegmentationClassAug | sed s/.png// > trainaug.txt
wget http://home.bharathh.info/pubs/codes/SBD/train_noval.txt  
mv train_noval.txt trainaug_noval.txt 
```

## Training

These examples use two gpus because we use slightly higher accuracy when we restricted the number of GPUs used. 

**Cityscapes Segmentation Training** with MobileNetV2 backbone and DeeplabV3Lite decoder can be done as follows:<br>
```
python ./references/pixel2pixel/train_segmentation_main.py --model_name deeplabv3plus_mobilenetv2_tv_edgeailite --dataset_name cityscapes_segmentation --data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth --gpus 0 1
```

Cityscapes Segmentation Training with **RegNet800MF backbone and FPN decoder** can be done as follows:<br>
```
python ./references/pixel2pixel/train_segmentation_main.py --dataset_name cityscapes_segmentation --model_name fpn_aspp_regnetx800mf_edgeailite --data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 --pretrained https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906036/RegNetX-800MF_dds_8gpu.pyth
```

It is possible to use a **different image size**. For example, we trained for 1536x768 resolution by the following. (We used a smaller crop size compared to the image resize resolution to reduce GPU memory usage). <br>
```
python ./references/pixel2pixel/train_segmentation_main.py --model_name deeplabv3plus_mobilenetv2_tv_edgeailite --dataset_name cityscapes_segmentation --data_path ./data/datasets/cityscapes/data --img_resize 768 1536 --rand_crop 512 1024 --output_size 1024 2048 --pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth --gpus 0 1
```

Train **FPNPixel2Pixel model at 1536x768 resolution** (use 1024x512 crop to reduce memory usage):<br>
```
python ./references/pixel2pixel/train_segmentation_main.py --model_name fpn_aspp_mobilenetv2_tv_edgeailite --dataset_name cityscapes_segmentation --data_path ./data/datasets/cityscapes/data --img_resize 768 1536 --rand_crop 512 1024 --output_size 1024 2048 --pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth --gpus 0 1
```
 
**VOC Segmentation Training** can be done as follows:<br>
```
python ./references/pixel2pixel/train_segmentation_main.py --model_name deeplabv3plus_mobilenetv2_tv_edgeailite --dataset_name voc_segmentation --data_path ./data/datasets/voc --img_resize 512 512 --output_size 512 512 --pretrained https://download.pytorch.org/models/mobilenet_v2-b0353104.pth --gpus 0 1
```

## Validation
During the training, **validation** accuracy will also be printed. But to explicitly check the accuracy again with **validation** set, it can be done as follows (fill in the path to the pretrained model):<br>
```
python ./references/pixel2pixel/train_segmentation_main.py --phase validation --model_name deeplabv3plus_mobilenetv2_tv_edgeailite --dataset_name cityscapes_segmentation --data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 --pretrained ?????
```


## Inference
Inference can be done as follows (fill in the path to the pretrained model):<br>
```
python ./references/pixel2pixel/infer_segmentation_main.py --phase validation --model_name deeplabv3plus_mobilenetv2_tv_edgeailite --dataset_name cityscapes_segmentation_measure --data_path ./data/datasets/cityscapes/data --img_resize 384 768 --output_size 1024 2048 --gpus 0 1 --pretrained ?????
```

 
## Results

|Dataset   |Mode Name                     |Input Size |GigaMACs  |mIoU Accuracy% |Model Name       |Notes |
|----------|------------------------------|-----------|----------|---------------|-----------------|------|
|           |**ADE20K32 dataset models**
|ADE20K32   |MobileNetV2S16+DeepLabV3PlusEdgeAILite |512x512    |3.28      |51.01          |                |      | 
|ADE20K32   |MobileNetV2+UNetEdgeAILite         |512x512    |2.427     |49.95          |                |      | 
|ADE20K32   |MobileNetV2+FPNEdgeAILite          |512x512    |3.481     |50.72          |                |      | 
|ADE20K32   |MobileNetV2-1.4+FPNEdgeAILite      |512x512    |6.646     |52.93          |                |      | 
|- 
|ADE20K32   |RegNetX400MF+FPNEdgeAILite         |384x384    |3.1526    |51.03          |                |      | 
|ADE20K32   |RegNetX800MF+FPNEdgeAILite         |512x512    |8.0683    |53.29          |                |      | 
|           |**COCOSeg21 dataset models**
|COCOSeg21  |MobileNetV2S16+DeepLabV3PlusEdgeAILite |512x512    |3.161     |57.77          |                |      | 
|COCOSeg21  |MobileNetV2+UNetEdgeAILite         |512x512    |2.009     |57.01          |                |      | 
|COCOSeg21  |MobileNetV2+FPNEdgeAILite          |512x512    |3.357     |               |                |      | 
|COCOSeg21  |RegNetX800MF+FPNEdgeAILite         |512x512    |7.864     |61.15          |                |      | 
|           |**Cityscapes dataset models**
|Cityscapes|RegNetX800MF+FPNEdgeAILite          |768x384    |**8.84**  |**72.01**      |fpn_aspp_regnetx800mf_edgeailite  |      | 
|Cityscapes|RegNetX1.6GF+FPNEdgeAILite          |1024x512   |**24.29** |**75.84**      |fpn_aspp_regnetx1p6gf_edgeailite  |      | 
|Cityscapes|RegNetX3.2GF+FPNEdgeAILite          |1024x512   |**49.40** |**77.24**      |fpn_aspp_regnetx3p2gf_edgeailite  |      | 
|Cityscapes|RegNetX3.2FF+FPNEdgeAILite          |1536x768   |**111.16**|**78.90**      |fpn_aspp_regnetx3p2gf_edgeailite  |      | 
|-
|Cityscapes|RegNetX400MF+FPNEdgeAILite          |768x384    |**6.09**  |**68.03**      |fpn_aspp_regnetx400mf_edgeailite  |      | 
|Cityscapes|RegNetX400MF+FPNEdgeAILite          |1536x768   |**24.37** |**73.96**      |fpn_aspp_regnetx400mf_edgeailite  |      | 
|-
|Cityscapes|MobileNetV2S16+DeepLabV3PlusEdgeAILite  |768x384    |**3.54**  |**69.13**      |deeplabv3plus_mobilenetv2_tv_edgeailite           |      | 
|Cityscapes|MobileNetV2+UNetEdgeAILite          |768x384    |**2.20**  |**68.94**      |unet_edgeailite_pixel2pixel_aspp_mobilenetv2_tv |      | 
|Cityscapes|MobileNetV2+FPNEdgeAILite           |768x384    |**3.84**  |**70.39**      |fpn_aspp_mobilenetv2_tv_edgeailite |      | 
|Cityscapes|MobileNetV2+FPNEdgeAILite           |1536x768   |**15.07** |**74.61**      |fpn_aspp_mobilenetv2_tv_edgeailite |      | 
|-
|Cityscapes|FD-MobileNetV2+FPNEdgeAILite        |1536x768   |**3.96**  |**71.28**      |fpn_aspp_mobilenetv2_tv_fd_edgeailite |      | 
|Cityscapes|FD-MobileNetV2+FPNEdgeAILite        |2048x1024  |**7.03**  |**72.67**      |fpn_aspp_mobilenetv2_tv_fd_edgeailite |      | 
|-
|Cityscapes |MobileNetV2S16+DeepLabV3PlusEdgeAILite-QAT* |768x384  |**3.54**  |**68.77**   |                |      |
|Cityscapes |MobileNetV2+UNetEdgeAILite-QAT*         |768x384  |**2.20**  |**68.18**   |                |      |
|Cityscapes |MobileNetV2+FPNEdgeAILite-QAT*          |768x384  |**3.84**  |**69.88**   |                |      |
|-
|Cityscapes|MobileNetV2+DeepLab[10]       |769x769    |21.27     |70.71          |                |      | 
|Cityscapes|MobileNetV3+DeepLab[10]       |769x769    |15.95     |72.41          |                |      | 
|Cityscapes|Xception65+DeepLab[10]        |769x769    |418.64    |78.79          |                |      | 
|Cityscapes|ERFNet[8]                     |1024x512   |27.705    |69.7           |                |      | 
|Cityscapes|MobileNetV2+SwiftNet[9]       |2048x1024  |41.0      |75.3           |                |      | 
|Cityscapes|ResNet50+FCN[3][11]           |1040x520   |285.4     |71.6           |                |      | 
|Cityscapes|ResNet50+DeepLabV3[5][11]     |1040x520   |337.5     |73.5           |                |      | 

*- Quantization Aware Training using 8b precision


#### Notes
- The suffix 'Lite' in the name of models such as DeepLabV3PlusEdgeAILite, FPNEdgeAILite & UNetEdgeAILite indicates the use of Depthwise convolutions or Grouped convolutions. If the feature extractor (encoder) uses Depthwise Convolutions, then Depthwise convolutions are used throughout such models, even in the neck and decoder. If the feature extractor (encoder) uses grouped convolutions as in the case of RegNetX, then grouped convolutions (with the same group size as that of the feature extractor) are used even in the neck and decoder.<br>
- GigaMACS: Complexity in Giga Multiply-Accumulations (lower is better). This is an important metric to watch out for when selecting models for embedded inference.<br>
- mIoU Accuracy%: Original Floating Point Validation Mean IoU Accuracy% obtained after training.<br>
- Overall, RegNetX based models strike a good balance between complexity, accuracy and easiness of quantization.<br>


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

[11] Torchvision & Torchvision Models, https://github.com/pytorch/vision/tree/master/torchvision, https://pytorch.org/docs/stable/torchvision/models.html

[12] FD-MobileNet: Improved MobileNet with a Fast Downsampling Strategy, Zheng Qin, Zhaoning Zhang, Xiaotao Chen, Yuxing Peng - https://arxiv.org/abs/1802.03750)
