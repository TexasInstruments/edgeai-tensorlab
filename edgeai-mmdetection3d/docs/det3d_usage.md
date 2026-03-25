# EdgeAI-MMDetection3D Usage

## Training/Evaluation/Testing

Refer the MMDetection3D documentation [Test and Train with Standard Datasets](./en/user_guides/train_test.md) for general floating point training/evaluation/testing steps for standard datasets. Use the below steps for training and evaluation without QAT.

### Steps for training and evaluation for pointpillars

1. cd to installation directory <install_dir>/edgeai-mmdetection3d

2. Do floating-model training using the command 
    "./tools/dist_train.sh configs/pointpillars/tidl_pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py <num_gpus>"

    For example, to use 2 GPUs use the command
    ```bash
    ./tools/dist_train.sh configs/pointpillars/tidl_pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py 2
    ```

3.  Do evalution using the command 

    "python ./tools/test.py configs/pointpillars/tidl_pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py <latest.pth file generated from previous step #2>" 

    For example,

    ```bash
    python ./tools/test.py ./configs/pointpillars/tidl_pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py ./work_dirs/tidl_pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car/epoch_80.pth
    ```
    Note: This is single GPU evalution command. "./dist_test.sh" can be used for multiple GPU evalution process.


For QAT training, follow the below steps.

### Steps for QAT training and evaluation for pointpillars

For QAT training follow the below steps.

1. cd to installation directory <install_dir>/edgeai-mmdetection3d


2. First do float training using the command 
    "./tools/dist_train.sh configs/pointpillars/tidl_pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py <num_gpus>"

    For example, to use 2 GPUs use the command
    ```bash
    ./tools/dist_train.sh configs/pointpillars/tidl_pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py 2
    ```

3. Do QAT training with loading the weights from previous step using the command 
    "./tools/dist_train.sh configs/pointpillars/tidl_pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car_qat.py <num_gpus>"

    For example,
    ```bash
    ./tools/dist_train.sh configs/pointpillars/tidl_pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car_qat.py 2
    ```

    Note: Check the load_from option in cfg file. It tries to load weights from previously trained model. If that path is not correct then change the cfg accordingly.

4.  Do Evalution using the command 

    "python ./tools/test.py configs/pointpillars/tidl_pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car_qat.py <latest.pth file generated from previous step #3>" 

    For example,

    ```bash
    python ./tools/test.py ./configs/pointpillars/tidl_pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car_qat_qat.py ./work_dirs/tidl_pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car_qat/epoch_80.pth
    ```

    Note: This is single GPU evalution command. "./dist_test.sh" can be used for multiple GPU evalution process.


## ONNX & Prototxt Export
Make sure that the appropriate config file is selected while evalution and set the field "save_onnx_model = True" in that config file, then onnx file and the prototxt file will be saved at the reload checkpoint location. This flow is supported for single GPU mode only.

Note: If you did QAT, then the flag quantize in the config file must be set to True even at this stage. 

