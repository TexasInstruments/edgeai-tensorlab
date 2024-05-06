# Depth Estimation

Depth Estimation predicts the depth (in the 3D space) associated with each pixel of the image. It is useful for tasks such as 3D Object detetion, surround view image stitching etc.

We suggest you to first read the documentation on [Sematic Segmentation](./Semantic_Segmentation.md) and also try out some examples in [run_edgeailite_segmentation.sh](../../run_edgeailite_segmentation.sh) before attempting Depth Estimation training.

In these examples we demonstrate the use of KITTI dataset for depth estimation training. We also support Cityscapes dataset, but we do not have an example for it here.

Commonly used Training/Validation commands are listed in the file [run_depth.sh](../../run_depth.sh). Uncommend one line and run the file to start the run.

Loss functions and many other parametes can be changed or configured in [references/edgeailite/scripts/train_depth_main.py](../../references/edgeailite/scripts/train_depth_main.py). We have seen that a combination of SmoothL1, ErrorVariance and Overall Scale Difference produces good results.

### Results

##### KITTI Depth Dataset

|Dataset    |Mode Architecture         |Backbone Model |Backbone Stride|Resolution |Complexity (GigaMACS)|ARD       |%ARD      |Model Configuration Name                  |
|---------  |----------                |-----------    |-------------- |-----------|--------             |----------|----------|----------------------------------------  |
|KITTI Depth|DeepLabV3PlusEdgeAILite with DWASPP |MobileNetV2    |16             |768x384    |**3.44**             |0.0705    |**7.05**  |**deeplabv3plus_mobilenetv2_tv_edgeailite**          |
|KITTI Depth|DeepLabV3PlusEdgeAILite with DWASPP |ResNet50       |32             |768x384    |**28.52**            |0.0631    |**6.31**  |**fpn_edgeailite_aspp_resnet50**         |

- ARD: Absolute Relative Difference<br>
- %ARD: Percentage Absolute Relative Difference