# Pre-complied / pre-imported model artifacts

This package provides Pre-Compiled Model Artifacts for several Deep Neural Network models. 
- URLs of Pre-compiled model artifacts can be obtained [in this folder](../modelartifacts/)

## These artifacts can be used for inference in multiple ways: 
- For inference/benchmark using [edgeai-benchmark](https://github.com/TexasInstruments/edgeai-benchmark) in PC or EVM
- In EdgeAI SDKs of the [supported SoCs](https://github.com/TexasInstruments/edgeai/blob/main/readme_sdk.md)


Note: The tasks performed by the pre-compiled model artifacts can be identified using the following fields in their names:
- cl: Image Classification Models
- od: Object Detection Models (COCO Object detection, Face detection)
- ss: Semantic Segmentation Models
- kd: Keypoint Detection Models (Human Pose Estimation)
- de: Depth Estimation
- lidar-3dod: 3D Object detection using LIDAR input
- 6d-pose: 6D Pose estimation of objects (3-Translation, 3-Rotation)
