
### Latest Model surgery tool (v2) 
[edgeai_torchmodelopt.xmodelopt.surgery.v2](../edgeai_torchmodelopt/xmodelopt/surgery/v2) - Easily replace layers (Modules, operators, functional) which could also include SOC specific unsupported layers with other layers without modifying the model code to create embedded friendly models. This uses torch.fx based surgery to handle models that uses torch modules, operators and functionals. (Compared to this, legacy surgery using **torch.nn** can only handle modules)<br>

The detailed usage and adding the custom replacement dictionary for the API is documented in [Model Surgery](../edgeai_torchmodelopt/xmodelopt/surgery/v2/README.md).

### Legacy Model surgery tool (v1)
[edgeai_torchmodelopt.xmodelopt.surgery.v1](../edgeai_torchmodelopt/xmodelopt/surgery/v1) - Our legacy implementation of Model surgery using **torch.nn** modules.<br>

The detailed usage and adding the custom replacement dictionary for the API is documented in [Model Surgery](../edgeai_torchmodelopt/xmodelopt/surgery/v1/README.md).

## Results

We use the default dictionary for model surgery. Here are the classification model results. The models that we obtain after the model surgery are called as lite models.

| Models        | Torchvision Accuracy          | Lite Model Accuracy   |
| ------------- |:-------------:    | :-----:                |
| MobileNet_V2   | 72.154         |   72.88                 |
| MobileNet_V3_Large  | 75.274    | 71.7*                 |
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

