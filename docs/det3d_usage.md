# EdgeAI-MMDetection3D Usage

## Training/Evaluation/Testing

Refer mmdetection3d documentation [Inference and Train with standard dataset](./en/1_exist_data_model.md) for training/evaluation/testing steps for standard dataset.

### Steps for QAT training for pointpillars
1. cd to installation directory <install_dir>/edgeai-mmdetection3d


2. First do normal training using the command 
    "./tools/dist_test.sh configs/pointpillars/tidl_hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py"


3. Do QAT training with loading the weights from previous step using the command 
    "./tools/dist_test.sh configs/pointpillars/tidl_hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_qat.py"


4.  Do Evalution using the command 
    "python ./tools/test.py tidl_hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_qat.py" 


5. To save TIDL friendly onnx file and related prototxt file set the flag "save_onnx_model" = True in the config file "tidl_hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_qat.py" and repeat the previous step of evaluation. 


## ONNX & Prototxt Export
Make sure that the appropriate config file is selected while training/evalution and set the field "save_onnx_model = True" in that config file, then onnx file and the prototxt file will be saved at the reload checkpoint location. This flow is supported for single GPU mode only.

Note: If you did QAT, then the flag quantize in the config file must be set to True even at this stage. 

