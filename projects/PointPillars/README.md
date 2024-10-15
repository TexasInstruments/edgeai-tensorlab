# PointPillars: Fast Encoders for Object Detection from Point Clouds

> [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)

<!-- [ALGORITHM] -->

## Abstract

Object detection in point clouds is an important aspect of many robotics applications such as autonomous driving. In this paper we consider the problem of encoding a point cloud into a format appropriate for a downstream detection pipeline. Recent literature suggests two types of encoders; fixed encoders tend to be fast but sacrifice accuracy, while encoders that are learned from data are more accurate, but slower. In this work we propose PointPillars, a novel encoder which utilizes PointNets to learn a representation of point clouds organized in vertical columns (pillars). While the encoded features can be used with any standard 2D convolutional detection architecture, we further propose a lean downstream network. Extensive experimentation shows that PointPillars outperforms previous encoders with respect to both speed and accuracy by a large margin. Despite only using lidar, our full detection pipeline significantly outperforms the state of the art, even among fusion methods, with respect to both the 3D and bird's eye view KITTI benchmarks. This detection performance is achieved while running at 62 Hz: a 2 - 4 fold runtime improvement. A faster version of our method matches the state of the art at 105 Hz. These benchmarks suggest that PointPillars is an appropriate encoding for object detection in point clouds.

<div align=center>
<img src="https://user-images.githubusercontent.com/79644370/143885905-aab6ffcf-7727-495e-90ca-edb8dd5e324b.png" width="800"/>
</div>

## Introduction

We implement PointPillars and provide the results and checkpoints on the KITTI dataset. The result can be found in [Object Detection Zoo](../../docs/det3d_modelzoo.md)

## Dataset Preperation
Prepare the KITTI dataset as per the MMDetection3D documentation [KITTI dataset preperation](../../docs/en/advanced_guides/datasets/kitti.md). 

**Note: Currently only KITTI dataset with pointPillars network is supported. For KITTI dataset optional ground plane data can be downloaded from [KITTI Plane data](https://download.openmmlab.com/mmdetection3d/data/train_planes.zip). For preparing the KITTI data with ground plane, please refer the MMDetection3D documentation [KITTI dataset preperation](../../docs/en/advanced_guides/datasets/kitti.md) and use below command from there**

### Steps for Kitti Dataset preperation
```bash
# Creating dataset folders
cd <edgeai-mmdetection3d>
mkdir ./data/kitti/ && mkdir ./data/kitti/ImageSets

# Download data split
wget -c https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/test.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/train.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/val.txt
wget -c  https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt --no-check-certificate --content-disposition -O ./data/kitti/ImageSets/trainval.txt

# Preparing the dataset
cd <edgeai-mmdetection3d>
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti --with-plane
```

### Steps for Semantic segmented painted Kitti Dataset preperation
PointPainting : It is simple fusion algorithm for 3d object detection. This repository supports data preperation and training for points painting. Please refer [PointPainting](https://arxiv.org/abs/1911.10150) for more details. Data preperation for points painting network is mentioned below

It is expected that previous step of normal KITTI data preperation is already done. Also required to do one small change in mmseg installation package. Please change the function "simple_test" in the file ~/.pyenv/versions/3.10.13/envs/mmdet3d/lib/python3.10/site-packages/mmseg/models/segmentors/encoder_decoder.py as shown below to return the segmentation output just after CNN network and before post processing. Please note that only return tensor is changed.
```python
def predict(self,
            inputs: Tensor,
            data_samples: OptSampleList = None) -> SampleList:
    """Predict results from a batch of inputs and data samples with post-
    processing.

    Args:
        inputs (Tensor): Inputs with shape (N, C, H, W).
        data_samples (List[:obj:`SegDataSample`], optional): The seg data
            samples. It usually includes information such as `metainfo`
            and `gt_sem_seg`.

    Returns:
        list[:obj:`SegDataSample`]: Segmentation results of the
        input images. Each SegDataSample usually contain:

        - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
        - ``seg_logits``(PixelData): Predicted logits of semantic
            segmentation before normalization.
    """
    if data_samples is not None:
        batch_img_metas = [
            data_sample.metainfo for data_sample in data_samples
        ]
    else:
        batch_img_metas = [
            dict(
                ori_shape=inputs.shape[2:],
                img_shape=inputs.shape[2:],
                pad_shape=inputs.shape[2:],
                padding_size=[0, 0, 0, 0])
        ] * inputs.shape[0]

    seg_logits = self.inference(inputs, batch_img_metas)

    # return seg_logits without postprocess
    #return self.postprocess_result(seg_logits, data_samples)
    return seg_logits
```
```bash
cd <edgeai-mmdetection3d>/tools/data_converter
python kitti_painting.py

cd <edgeai-mmdetection3d>
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti_point_painting --with-plane
```

## Get Started
Please see [Usage](../../docs/det3d_usage.md) for training and testing with this repository.


## 3D Object Detection Model Zoo
Complexity and Accuracy report of several trained models is available at the [3D Detection Model Zoo](../../docs/det3d_modelzoo.md) 


## Quantization
This tutorial explains more about quantization and how to do [Quantization Aware Training (QAT)](../../docs/det3d_quantization.md) of detection models.


## ONNX & Prototxt Export
**Export of ONNX model (.onnx) and additional meta information (.prototxt)** is supported. The .prototxt contains meta information specified by **TIDL** for object detectors. 

The export of meta information is now supported for **PointPillars** detectors. 

For more information please see [Usage](../../docs/det3d_usage.md).

## References

[1] PointPillars: Fast Encoders for Object Detection from Point Clouds, A. H. Lang, S. Vora, H. Caesar, L. Zhou, J. Yang, O. Beijbom, https://arxiv.org/abs/1812.05784

[2] PointPainting: Sequential Fusion for 3D Object Detection, S. Vora, A. H. Lang, B. Helou, O. Beijbom, https://arxiv.org/abs/1911.10150