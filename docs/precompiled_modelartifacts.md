# Pre-Complied / Pre-Imported Model Artifacts

This package provides Pre-Compiled Model Artifacts for several Deep Neural Network models. 

URLs of Pre-Compiled model artifacts can be obtained [in this folder](../modelartifacts/8bits/) and information is provided in this [list](../modelartifacts/8bits/artifacts.list)

Note: To run inference with these artifacts, there is no need to download them manually - they will be downloaded automatically during the inference using the URLs in the .link files.

Note: These artifacts can be used for inference in multiple ways: 
- [Jypyter Notebook examples in TIDL](https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/latest/exports/docs/tidl_j7_02_00_00_07/ti_dl/docs/user_guide_html/md_tidl_notebook.html) 
- For inference/benchmark in this Jacinto-AI-Benchmark repository 
- In [EdgeAI Cloud Evaluation](https://dev.ti.com/edgeai/) 
- In EdgeAI-SDK (to be announced)

The tasks performed by the pre-compiled model artifacts can be identified using the following fields in their names:
- cl: Image Classification Models
- od: Object Detection Models
- ss: Semantic Segmentation Models
- kd: Keypoint Detection Models (Human Pose Estimation)
