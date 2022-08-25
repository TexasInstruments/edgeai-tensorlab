# EdgeAI-MMDetection3D Usage

## Training/Evaluation/Testing
Refer mmdetection3d documentation [mmdetection3d](README_mmdet3d.md) for training/evaluation/testing steps with approriate config file selected.

## ONNX & Prototxt Export
Make sure that the appropriate config file is selected while training/evalution and set the field "save_onnx_model = True" in that config file, then onnx file and the prototext file will be saved at the reload checkpoint location. This flow is supported for single GPU mode only.

Note: If you did QAT, then the flag quantize in the config file must be set to True even at this stage. 

