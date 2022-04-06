# Pre-complied / pre-imported model artifacts

This package provides Pre-Compiled Model Artifacts for several Deep Neural Network models. 
- URLs of Pre-compiled model artifacts can be obtained [in this folder](../modelartifacts/8bits/)
- Summary information is provided in this [list](../modelartifacts/8bits/artifacts.csv)
- Accuracy Report is [here](../modelartifacts/report_accuracy.csv)

Note: These artifacts can be used for inference in multiple ways: 
- For inference/benchmark in [edgeai-benchmark](https://github.com/TexasInstruments/edgeai-benchmark)
- In EdgeAI SDK StarterKit **[PROCESSOR-SDK-LINUX-SK-TDA4VM](https://www.ti.com/tool/download/PROCESSOR-SDK-LINUX-SK-TDA4VM)**
- In [EdgeAI Cloud Evaluation](https://dev.ti.com/edgeai/) 


The tasks performed by the pre-compiled model artifacts can be identified using the following fields in their names:
- cl: Image Classification Models
- od: Object Detection Models (COCO Object detection, Face detection)
- ss: Semantic Segmentation Models
- kd: Keypoint Detection Models (Human Pose Estimation)
- de: Depth Estimation
- lidar-3dod: 3D Object detection using LIDAR input
