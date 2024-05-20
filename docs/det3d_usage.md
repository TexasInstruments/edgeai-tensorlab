# EdgeAI-MMDetection3D Usage

## Training/Evaluation/Testing

Refer mmdetection3d documentation [Inference and Train with standard dataset](./en/1_exist_data_model.md) for floating point training/evaluation/testing steps for standard dataset. For QAT training floow the below steps.

### Steps for QAT training for pointpillars
1. cd to installation directory <install_dir>/edgeai-mmdetection3d


2. First do float training using the command 
    "./tools/dist_train.sh configs/pointpillars/tidl_hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py <num_gpus>"

    E.g. to use 2 GPUs use the command
```bash
    ./tools/dist_train.sh configs/pointpillars/tidl_hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py 2
```

3. Do QAT training with loading the weights from previous step using the command 
    "./tools/dist_train.sh configs/pointpillars/tidl_hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_qat.py <num_gpus>"
    
    E.g. 

```bash
    ./tools/dist_train.sh configs/pointpillars/tidl_hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_qat.py 2
```

    Note: Check the load_from option in cfg file. It tries to load weights from previously trained model. If that path is not correct then change the cfg accordingly.


4.  Do Evalution using the command 

    "python ./tools/test.py tidl_hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_qat.py <latest.pth file generated from previous step #3> --eval mAP" 

    e.g.

```bash
    python ./tools/test.py ./configs/pointpillars/tidl_hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_qat.py ./work_dirs_quant/tidl_hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_qat/latest.pth
```
    Note: This is single GPU evalution command. "./dist_test.sh" can be used for multiple GPU evalution process.


## ONNX & Prototxt Export
Make sure that the appropriate config file is selected while evalution and set the field "save_onnx_model = True" in that config file, then onnx file and the prototxt file will be saved at the reload checkpoint location. This flow is supported for single GPU mode only.

Note: If you did QAT, then the flag quantize in the config file must be set to True even at this stage. 

