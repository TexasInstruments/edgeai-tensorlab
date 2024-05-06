# Efficientdetlite Object Detection Models

## Introduction
* Efficientdet family of object detectors are based on efficientnet backbone and Bi_FPN feature extractor.<br>

* Efficientnet-lite is lighter version of efficientnet for edge applications. Similarly, we have come up with a lighter version of efficientdet that we call Efficientdet-ti-lite. This naming convention is chosen to avoid conflict with future release of efficientde-lite models from Google.

- Following changes were made to get efficientdet-ti-lite from efficientdet:
    * Remove Squeeze and excitations.
    * Replace swish activations with ReLU.
    * Fast normalization feature fusion replaced with Simple addition.
    * (K=3,S=2) max-pooling replaced with (k=2,s=2) max-pooling.
    * All max-pooling has a ReLU before it.
    * During inference, d/w convolutions(k=5.s=2) were replaced with two d/w convolutions {(k=5,s=1), (k=3,s=1)}

-  Training logs are available in the current directory itself.

|Dataset |Model Name                              |Input Size |GigaMACS  |AP[0.5:0.95]%, AP50%|Notes |
|--------|----------------------------------------|-----------|----------|--------------------|----- |
|        |**Our training - Lite models**
|COCO    |EfficientDet-Lite0_BiFPN_MaxPool2x2_ReLU|512x512    |**2.50**  |33.61, 52.27        |ti-lite flavour|
|COCO    |EfficientDet-Lite1_BiFPN_MaxPool2x2_ReLU|640x640    |**5.49**  |                    |ti-lite flavour|
|COCO    |EfficientDet-Lite2_BiFPN_MaxPool2x2_ReLU|768x768    |**9.96**  |                    |ti-lite flavour|
|COCO    |EfficientDet-Lite3_BiFPN_MaxPool2x2_ReLU|896x896    |          |                    |ti-lite flavour|
|          |**Model from original repository**
|COCO      |EfficientDet-D0                       |512x512    |          |34.3, 53.0          |      |


## References

[1] EfficientDet: Scalable and Efficient Object Detection, Mingxing Tan, Ruoming Pang, Quoc V. Le, https://arxiv.org/abs/1911.09070 
[2] Brain AutoML, github.com/google/automl, https://github.com/google/automl
[3] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, Mingxing Tan, Quoc V. Le, https://arxiv.org/abs/1905.11946
[4] Higher accuracy on vision models with EfficientNet-Lite, Renjie Liu, https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html
