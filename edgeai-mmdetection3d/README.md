# EdgeAI-MMDetection3D



This repository is an extension of the popular [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) open source repository for 3d object detection. While mmdetection3d focuses on a wide variety of models, typically at high complexity, we focus on models that are optimized for speed and accuracy so that they run efficiently on embedded devices. For this purpose, we have added a set of embedded friendly model configurations and scripts.

This repository also supports Quantization Aware Training (QAT).

<hr>


## Release Notes
See notes about recent changes/updates in this repository in [release notes](./docs/det3d_release_notes.md)


## Installation Steps for QAT Training
Follow installation steps of [edgeai-torchvision](https://github.com/TexasInstruments/edgeai-torchvision) to install edgeai-torchvision. In same environment install [mmdetection3d](README_mmdet3d.md) by skipping pytorch and CUDA installtion step mentioned in the [mmdetection3d Installation Guide](./docs/en/getting_started.md#installation). As pytorch and CUDA must have been installed as part of edgeai-torchvision installation.

Steps #1 create working conda environment 
```bash
conda create python=3.7 -n mmdet3d
conda activate mmdet3d
```
Steps #2 Install edgeai-torchvision
```bash
git clone https://github.com/TexasInstruments/edgeai-torchvision.git
cd <edgeai-torchvision>
./setup.sh
```

Steps #3 edgeai-mmdetection3d
```bash
git clone https://github.com/TexasInstruments/edgeai-mmdetection3d.git
cd <edgeai-mmdetection3d>
pip3 install openmim
mim install mmcv-full==1.5.2
mim install mmdet==2.25.0
mim install mmsegmentation==0.27.0
pip3 install -e .
install other additional requirements by typing 
pip install -r ./requirements.txt
pip install setuptools==59.5.0
```

Note: It may ask to downgrade protobuf. In that case uninstall exisitng protobuf and downgrade to 3.9.0 using the command 
```bash
pip uninstall protobuf
pip install protobuf==3.9.0
```

## Dataset Preperation
Prepare dataset as per original mmdetection3d documentation [dataset preperation](./docs/en/data_preparation.md). 

**Note: Currently only KITTI dataset with pointPillars network is supported. For KITTI dataset optional ground plane data can be downloaded from [KITTI Plane data](https://download.openmmlab.com/mmdetection3d/data/train_planes.zip). For preparing the KITTI data with ground plane, please refer the mmdetection3d external link [dataset preperation external Link](https://mmdetection3d.readthedocs.io/en/latest/datasets/kitti_det.html) and use below command from there**

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
PointPainting : It is simple fusion algorithm for 3d object detection. This repository supports data preperation and training for points painting. Please refer https://arxiv.org/abs/1911.10150 for more details. Data preperation for points painting network is mentioned below

It is expected that previous step of normal KITTI data preperation is already done. Also required to do one small change in mmseg installation package. Please change the function "simple_test" in the file ~/anaconda3/envs/<conda env name>/lib/python3.7/site-packages/mmseg/models/segmentors/encoder_decoder.py as shown below to return the segmentation output just after CNN network and before argmax. Please note that only return tensor is changed.
```python
def simple_test(self, img, img_meta, rescale=True):
    """Simple test with single image."""
    seg_logit = self.inference(img, img_meta, rescale)
    seg_pred = seg_logit.argmax(dim=1)
    if torch.onnx.is_in_onnx_export():
        # our inference backend only support 4D output
        seg_pred = seg_pred.unsqueeze(0)
        return seg_pred
    seg_pred = seg_pred.cpu().numpy()
    # unravel batch dim
    seg_pred = list(seg_pred)
    return seg_logit # changed from "return seg_pred"
```
```bash
cd <edgeai-mmdetection3d>/tools/data_converter

python kitti_painting.py
```

## Get Started
Please see [Usage](./docs/det3d_usage.md) for training and testing with this repository.


## 3D Object Detection Model Zoo
Complexity and Accuracy report of several trained models is available at the [3D Detection Model Zoo](./docs/det3d_modelzoo.md) 


## Quantization
This tutorial explains more about quantization and how to do [Quantization Aware Training (QAT)](./docs/det3d_quantization.md) of detection models.


## ONNX & Prototxt Export
**Export of ONNX model (.onnx) and additional meta information (.prototxt)** is supported. The .prototxt contains meta information specified by **TIDL** for object detectors. 

The export of meta information is now supported for **pointPillars** detectors.

For more information please see [Usage](./docs/det3d_usage.md)


## Advanced documentation
Kindly take time to read through the documentation of the original [mmdetection3d](README_mmdet3d.md) before attempting to use extensions added to this repository.


 
## Acknowledgement

This is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.

We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to train existing detectors and also to develop their own new detectors.


## License

Please see [LICENSE](./LICENSE) file of this repository.


## Model deployment

Now MMDeploy has supported some MMDetection3D model deployment. Please refer to [model_deployment.md](docs/en/tutorials/model_deployment.md) for more details.

## Citation

This package/toolbox is an extension of mmdetection3d (https://github.com/open-mmlab/mmdetection3d). If you use this repository or benchmark in your research or work, please cite the following:

```
@article{EdgeAI-MMDetection3D,
  title   = {{EdgeAI-MMDetection3D}: An Extension To Open MMLab Detection Toolbox and Benchmark},
  author  = {Texas Instruments EdgeAI Development Team, edgeai-devkit@list.ti.com},
  journal = {https://github.com/TexasInstruments/edgeai},
  year={2022}
}
```

```
@misc{mmdet3d2020,
    title={{MMDetection3D: OpenMMLab} next-generation platform for general {3D} object detection},
    author={MMDetection3D Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmdetection3d}},
    year={2020}
}
```

## References
[1] MMDetection3d: https://github.com/open-mmlab/mmdetection3d

[2] PointPillars: https://arxiv.org/abs/1812.05784

[3] PointPainting: https://arxiv.org/abs/1911.10150