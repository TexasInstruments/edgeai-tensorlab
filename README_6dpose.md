# YOLO-6D-Pose Multi-Object 6D Pose Estimation Model
This repository is the official implementation of the paper ["**YOLO-6D-Pose: Enhancing YOLO for Multi Object 6D Pose Estimation**"](https://arxiv.org/abs/2204.06806).It contains YOLOX based models for 6D Object pose estimation.
This repository is based on the YOLOX training and assumes that all dependencies for training YOLOX are already installed.

Additional requirements can be installed with the command below:
```
pip install -r requirements_6d.txt
```
Given below is a sample inference with ground-truth pose in green and predicted pose overlayed with class-specific colors.
<br/> 
<p float="left">
<img width="600" src="./assets/demo_6d.png">
</p>   

## **Datset Preparation**
The dataset needs to be prepared in YOLO format so that the enhanced dataloader can read the 6D pose along with the bounding box for each object. We currently support YCBV 
and LINEMOD datset.
### **YCBV Datset**
THe following components needs to be downloaded and structrured in the required way inside **edgeai_yolox/datasets** for the dataloader to read it correctly :
* [Base archive](https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_base.zip)
* [models](https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_models.zip), 524MB 
* [train_pbr](https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_train_pbr.zip), 21GB
* [train_real](https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_train_real.zip), 75.7GB
* [All test images](https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_test_all.zip), 15GB
* [BOP test images](https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_test_bop19.zip), 660MB

All required components for YCBV dataset can be downloaded with the script below. This will structure them in the required format as well.
```
./download_ycbv.sh
```
Once downloaded, the dataset for a given split has to be converted to COCO fromat with the script below:
```
python tools/ycb2coco.py --datapath './datasets/ycbv' --split train 
                                                      --split test                   # 900 frames for testing are used by default as in BOP format                                                                    
```
The above script will generate **instances_train.json** (), **instances_test.json** () and **instances_test_bop.json** ().
* **instances_train.json**: Contains annotations for all **50K** pbr images. From the set of real images, we select every 10th frame, resulting in **11355** real images. 
    In total, there are **61355** frames in the training set.
* **instances_test.json** Contains annotations for **2949** default test images.
* **instances_test_bop.json** Contains annotations for **900** test images used for BOP evaluation.

Expected directory structure:
```
edgeai-yolox
│   README.md
│   ...   
|   datasets
|     ycbv
│        annotations
│         └───instances_test.json
│         └───instances_test_bop.json
│         └───instances_train.json 
│        base
|        models
|        models_eval
|        models_fine
|        test_bop
|        test_real
|        train_pbr
│         └───000000
│         └───000001
|         .
|         .
│         └───000091
|        train_real
│         └───000000
│         └───000001
|         .
|         .
│         └───000049
```
### **LINEMOD Occlusion Dataset**
SImilar steps as YCBV need to be followed for LINEMOD Occlusion dataset as well. Following files are required to be downloaded and extract in **edgeai_yolox/datasets**:.
* [Base archive](https://bop.felk.cvut.cz/media/data/bop_datasets/lmo_base.zip)
* [models](https://bop.felk.cvut.cz/media/data/bop_datasets/lmo_models.zip), 5.4MB
* [train_pbr](https://bop.felk.cvut.cz/media/data/bop_datasets/lm_train_pbr.zip), 21.8GB
* [All test images](https://bop.felk.cvut.cz/media/data/bop_datasets/lmo_test_all.zip), 720.2MB
* [BOP test images](https://bop.felk.cvut.cz/media/data/bop_datasets/lmo_test_bop19.zip) 117.6MB
* train_real #Not applicable for LineMOD dataset as there is no real training data available that contains annotation for all objects present in the image.

Download all required components for LMO dataset with the script below. This will structure the dataset in the required format as well.
```
./download_lmo.sh
```
In order to convert LINEMOD-Occlusion datset to COCO format, run the following command:
```
#This portion can be part of readme.
python tools/lm2ococo.py --datapath './datasets/lmo' --split train                
                                                     --split test    # 200 frames for testing are used by default as in BOP format   
```
The above script will generate **instances_train.json** (253.9MB) and **instances_test_bop.json** (429.6KB).
## **YOLO-6D-Pose Models and Ckpts**.

|Dataset | Model Name              |Input Size |GFLOPS | AR  | AR<sub>VSD</sub>| AR<sub>MSSD</sub> | AR<sub>MSPD</sub> | ADD(s)| Notes |
|--------|-------------------------|-----------|-------|-----|-----------------|-------------------|-------------------|-------|-------|
|YCBV    | [YOLOX_s_object_pose]() |640x480    | 31.2  | 67.1|     62.4        |      68.0         |      70.8         | 59.4  |[pretrained_weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth)|
|YCBV    | [YOLOX_m_object_pose]() |640x480    | 80.3  | 75.4|     71.0        |      76.7         |      78.4         | 71.1  |[pretrained_weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth)|
|YCBV    | [YOLOX_l_object_pose]() |640x480    | 161.2 | 81.1|     76.0        |      83.1         |      84.0         | 81.1  |[pretrained_weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth)|

## **Training: YOLO-6D-Pose**
Train a model  by running the command below. Pretrained ckpt for each model is the corresponding 2D object detection model trained on COCO dataset.

```
python -m  yolox.tools.train -n yolox-s-object-pose --dataset ycbv -c 'path to pretrained ckpt' -d 8 -b 64 --fp16 -o --task object_pose 
                                yolox-m-object-pose           lmo
                                yolox-l-object-pose           lm 
```
## **YOLOX-ti-lite 6D Pose Models and Ckpts**
This is a lite version of the the model as described here. These models will run efficiently on TI processors.

|Dataset |          Model Name            |Input Size |GFLOPS| AR  | AR<sub>VSD</sub>| AR<sub>MSSD</sub>|AR<sub>MSPD</sub>|ADD(s)| Notes |
|--------|------------------------------- |-----------|------|-----|-----------------|------------------|-----------------|------|-------|
|YCBV    |[YOLOX_s_object_pose_ti_lite]() |640x480    | 31.2 |66.0 |      60.8       |      67.3        |     70.0        | 53.8 |[pretrained_weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth)|
|YCBV    |[YOLOX_m_object_pose_ti_lite]() |640x480    | 80.4 |74.4 |      69.2       |      75.7        |     78.3        | 70.9 |[pretrained_weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth)|
|YCBV    |[YOLOX_l_object_pose_ti_lite]() |640x480    | 161.4|     |                 |                  |                 |      |[pretrained_weights]()|

## **Training: YOLO-6D-Pose-ti-lite**
Train a suitable model  by running the command below. Pretrained ckpts for these lite models are same as the original models.
```
python -m yolox.tools.train -n yolox-s-object-pose-ti-lite --dataset ycbv -c 'path to pretrained ckpt' -d 8 -b 64 --fp16 -o --task object_pose
                            -n yolox-m-object-pose-ti-lite           lmo              
                            -n yolox-l-object-pose-ti-lite           lm  
```

## **Model Testing** 
The eval script computes ADD(s) score for a model. Apart from that, it generates a CSV file **'bop_test.csv'**. This file can be used to generate **AR,  AR<sub>VSD</sub>, AR<sub>MSSD</sub>, AR<sub>MSPD</sub>** using [bop_toolkit](https://github.com/thodan/bop_toolkit) repo.

Run the following command to compute ADD(s) metric on a pretrained checkpoint:
  ```
  python -m yolox.tools.eval -n yolox-s-object-pose           --dataset ycbv  -b 64 -d 8 -c "path to ckpt" --task object_pose --fp16 --fuse
                                yolox-s-object-pose-ti-lite 
  ```
Now, use the csv file generated by the above scirpt to get all the BOP metrics. Clone the [bop_toolkit](https://github.com/thodan/bop_toolkit) repo and install the rquired dependencies. Final evaluation command is:
  ```
  python eval_bop19_pose.py "path to the csv file"
  ```
<br/> 

###  **ONNX Export Including Detection and 6D Pose Estimation:** #Working on the export. Export of the camera parameters need to be added.
* Run the following command to export the entire model including the detection and 6D pose estimation part,
    ``` 
    python tools/export_onnx.py -n yolox-s-object-pose          -c "path to ckpt"  --output-name yolox_s_object_pose.onnx         --export-det --dataset ycbv --task object_pose
                                   yolox-s-object-pose-ti-lite                                   yolox_s_object_pose_ti_lite.onnx
    ```
* Apart from exporting the complete ONNX model, above script will generate a prototxt file that contains information of the detection layer. This prototxt file is required to deploy and acclerate the moodel on TI SoC.

###  **ONNXRT Inference: 6D Object Pose Estimation Inference with an End-to-End ONNX Model:**
 * If you haven't exported a model with the above command, download a sample model from this [link]().
 * Run the script as below to run inference with an ONNX model. The script runs inference and visualize the results. 
    ``` 
    cd demo
    # Run inference on a set of sample images as specified by sample_ips.txt
    python onnx_inference_object_pose.py --model-path "path_to_onnx_model"  --img-path "sample_ips.txt" --dst-path "sample_ops_onnxrt"  
    ```
 * The ONNX model infers the six dimenaional rerpresentation of the rotation parameter. The rotation matrix is deterministically computed from this 6D representation. This is done outside the model.
 * The camera parameters for a given dataset are part of the ONNX model. The exact translation parameters are computed from it's 2D proejetcion. This is computed inside the model.
###  **ONNXRT Inference on TDA4X: 6D Object Pose Estimation Import and Inference with an End-to-End ONNX Model:**
* Similar to running the model on PC using ONNXRT, it can be deployed in TI SOC in a similar fashion.
* You can follow the installation instruction from [here]().
* Here, we need to compile the model first as shown below. This will convert the model to fix-point from floating point. This is done in PC.
    ``` 
    # Run compilation on a set of sample images as specified by calib.txt
    python onnx_inference_object_pose.py ---compile --model-path "path_to_onnx_model" --img-path "sample_ips.txt" --dst-path "sample_ops_onnxrt"  
    ```
* After compilation, the model can deployed on TI SOC for inference. For sanity check, one can run inference on PC as well. 
    ``` 
    # Run inference on a set of sample images as specified by sample_ips.txt
    python onnx_inference_object_pose.py ---inference --model-path "path_to_onnx_model" --img-path "sample_ips.txt" --dst-path "sample_ops_onnxrt"  
    ```
