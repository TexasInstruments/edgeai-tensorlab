# YOLO-6D-Pose Multi-Object 6D Pose Estimation Model
This repository is the official implementation of the paper ["**YOLO-6D-Pose: Enhancing YOLO for Multi Object 6D Pose Estimation**"]().It contains YOLOX based models for 6D Object pose estimation.

Given below is a sample inference with ground-truth pose in green and predicted pose overlayed with class-specific colors.
<br/> 
<p float="left">
<img width="600" src="./assets/demo_6d.png">
</p>   

## **Datset Preparation**
The dataset needs to be prepared in YOLO format so that the enhanced dataloader can read the 6D pose along with the bounding box for each object. We currently support YCBV 
and LINEMOD datset.
### **YCBV Datset**
The following components needs to be downloaded and structrured in the required way inside **edgeai_yolox/datasets** for the dataloader to read it correctly :
* [Base archive](https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_base.zip)
* [models](https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_models.zip), 524MB 
* [train_pbr](https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_train_pbr.zip), 21GB
* [train_real](https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_train_real.zip), 75.7GB
* [All test images](https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_test_all.zip), 15GB
* [BOP test images](https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_test_bop19.zip), 660MB

All required components for YCBV dataset can be downloaded with the script below. This will structure them in the required format as well. Please ensure that **~200GB** of space is available in **edgeai_yolox/datasets** before running this script.
```
./download_ycbv.sh
```
Once downloaded, the dataset for train and test split has to be converted to COCO format with the script below:
```
python tools/ycb2coco.py --datapath './datasets/ycbv' --split train 
                                                      --split test                   # 900 frames for testing are used by default as in BOP format
```
The above script will generate **instances_train.json** (313.9MB), **instances_test_bop.json** (1.6MB) and **instances_test_all.json** (5.5MB).
* **instances_train.json**: Contains annotations for all **50K** pbr images. From the set of real images, we select every 10th frame, resulting in **11355** real images. 
    In total, there are **61355** frames in the training set.
* **instances_test_bop.json** Contains annotations for **900** test images used for BOP evaluation. This split is used by default for evaluation.
* **instances_test_all.json** Contains annotations for **2949** default test images.


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
Similar steps as YCBV need to be followed for LINEMOD Occlusion dataset as well. Following files are required to be downloaded and extract in **edgeai_yolox/datasets**:.
* [Base archive](https://bop.felk.cvut.cz/media/data/bop_datasets/lmo_base.zip)
* [models](https://bop.felk.cvut.cz/media/data/bop_datasets/lmo_models.zip), 5.4MB
* [train_pbr](https://bop.felk.cvut.cz/media/data/bop_datasets/lm_train_pbr.zip), 21.8GB
* [All test images](https://bop.felk.cvut.cz/media/data/bop_datasets/lmo_test_all.zip), 720.2MB
* [BOP test images](https://bop.felk.cvut.cz/media/data/bop_datasets/lmo_test_bop19.zip) 117.6MB
* train_real #Not applicable for LineMOD dataset as there is no real training data available that contains annotation for all objects present in the image.

Download all required components for LMO dataset with the script below. This will structure the dataset in the required format as well. Please ensure that **~200GB** of space is available in **edgeai_yolox/datasets** before running this script.
```
./download_lmo.sh
```
In order to convert LINEMOD-Occlusion datset to COCO format, run the following command for the train and test split:
```
python tools/lmo2coco.py --datapath './datasets/lmo' --split train                
                                                     --split test    # 200 frames for testing are used by default as in BOP format
```
The above script will generate **instances_train.json** (136.6MB) and **instances_test_bop.json** (429.6KB).
* **instances_train.json**: Contains annotations for all **50K** pbr images. From the set of real images.
* **instances_test_bop.json** Contains annotations for **200** test images used for BOP evaluation. This split is used by default for evaluation.
## **YOLOX-6D-Pose Models and Ckpts**.

### **Models Trained on YCB-V**.
|Dataset | Model Name              |Input Size |GFLOPS |  Params(M) |AR  | AR<sub>VSD</sub>| AR<sub>MSSD</sub> | AR<sub>MSPD</sub> | ADD(s)| Pretrained Weights | weights|
|--------|-------------------------|-----------|-------|-----------|-----|-----------------|-------------------|-------------------|-------|-------|-------|
|PBR+Real   | [YOLOX_s_object_pose](./exps/default/yolox_s_object_pose.py) |640x480    | 31.2  | 11.6 | 70.77|  65.86  | 72.29   |   74.17  | 66.99  |[pretrained_weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth)| [weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/ycbv/edgeai-yolox/checkpoints/yolox_s_object_pose/best_ckpt.pth) |
|PBR+Real   | [YOLOX_m_object_pose](./exps/default/yolox_m_object_pose.py) |640x480    | 80.3  |31.3   | 76.30|  71.13  |  77.04 |   80.75 | 73.17  |[pretrained_weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth)| [weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/ycbv/edgeai-yolox/checkpoints/yolox_m_object_pose/best_ckpt.pth) |
|PBR+Real   | [YOLOX_l_object_pose](./exps/default/yolox_l_object_pose.py) |640x480    | 161.2 | 64.8   |81.31|  76.74 |  83.62  |  83.56   | 83.16  |[pretrained_weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth)| [weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/ycbv/edgeai-yolox/checkpoints/yolox_l_object_pose/best_ckpt.pth) |
|PBR+Real   | [YOLOX_x_object_pose](./exps/default/yolox_x_object_pose.py) |640x480    | 281.0 | 115.6  |83.49|  79.25  | 85.87  |  85.37 | 88.09  |[pretrained_weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth)| [weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/ycbv/edgeai-yolox/checkpoints/yolox_x_object_pose/best_ckpt.pth) |


### **Models Trained on LM-O**.
|Dataset | Model Name              |Input Size |GFLOPS |  Params(M)|AR  | AR<sub>VSD</sub>| AR<sub>MSSD</sub> | AR<sub>MSPD</sub> | ADD(s)| Pretrained Weights | weights|
|--------|-------------------------|-----------|-------|-----------|----|-----------------|-------------------|-------------------|-------|--------------------|--------|
| PBR  | [YOLOX_s_object_pose](./exps/default/yolox_s_object_pose.py) |640x480    | 31.2  | 11.6   | 56.13 |  40.21  |  51.18  |  77.00   | 27.23  |[pretrained_weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth)| [weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/lmo/edgeai-yolox/checkpoints/yolox_s_object_pose/best_ckpt.pth)|
| PBR  | [YOLOX_m_object_pose](./exps/default/yolox_m_object_pose.py) |640x480    | 80.3  | 31.3   | 59.43 |  44.13  |  56.24  |  77.93   | 36.36  |[pretrained_weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth)| [weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/lmo/edgeai-yolox/checkpoints/yolox_m_object_pose/best_ckpt.pt)|
| PBR  | [YOLOX_l_object_pose](./exps/default/yolox_l_object_pose.py) |640x480    | 161.2 | 64.8   | 61.51 |  46.29  |  58.52  |  79.71   | 39.39  |[pretrained_weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth)| [weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/lmo/edgeai-yolox/checkpoints/yolox_l_object_pose/best_ckpt.pth)|
| PBR  | [YOLOX_X_object_pose](./exps/default/yolox_x_object_pose.py) |640x480    | 281.0 | 115.6  | 62.90 |  48.25  |  61.09  |  79.37   | 44.47  |[pretrained_weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth)| [weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/lmo/edgeai-yolox/checkpoints/yolox_x_object_pose/best_ckpt.pth)|

## **Training: YOLOX-6D-Pose**
Train a model  by running the command below. Pretrained ckpt for each model is the corresponding 2D object detection model trained on COCO dataset.

```
python -m  yolox.tools.train -n yolox-s-object-pose --dataset ycbv -c 'path to pretrained ckpt' -d 4 -b 64 --fp16 -o --task object_pose 
                                yolox-m-object-pose           lmo
                                yolox-l-object-pose
                                yolox-x-object-pose
```
## **YOLOX-6D-Pose-ti-lite Models and Ckpts**
These are lite version of the the model as described [here](./README_2d_od.md). These models are optimized to run efficiently on TI processors.

### **Ti-lite Models Trained on YCB-V**.
|Dataset |          Model Name            |Input Size |GFLOPS| AR  | AR<sub>VSD</sub>| AR<sub>MSSD</sub>|AR<sub>MSPD</sub>|ADD(s)| Pretrained Weights |weights|
|--------|------------------------------- |-----------|------|-----|-----------------|------------------|-----------------|------|-------|-------|
|PBR+Real    |[YOLOX_s_object_pose_ti_lite](./exps/default/yolox_s_object_pose_ti_lite.py) |640x480    | 31.2 |66.0 |      60.8       |      67.3        |     70.0        | 53.8 |[pretrained_weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth)| [weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/ycbv/edgeai-yolox/checkpoints/yolox_s_object_pose_ti_lite/best_ckpt.pth)|
|PBR+Real    |[YOLOX_m_object_pose_ti_lite](./exps/default/yolox_m_object_pose_ti_lite.py) |640x480    | 80.4 |74.4 |      69.2       |      75.7        |     78.3        | 70.9 |[pretrained_weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth)| [weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/ycbv/edgeai-yolox/checkpoints/yolox_m_object_pose_ti_lite/best_ckpt.pth)|
|PBR+Real    |[YOLOX_l_object_pose_ti_lite](./exps/default/yolox_l_object_pose_ti_lite.py) |640x480    | 161.4|78.2 |      72.9       |      80.2        |     81.5        | 77.0|[pretrained_weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth)| [weights](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/ycbv/edgeai-yolox/checkpoints/yolox_l_object_pose_ti_lite/best_ckpt.pth)|

## **Training: YOLOX-6D-Pose-ti-lite**
Train a suitable model  by running the command below. Pretrained ckpts for these lite models are same as the original models.
```
python -m yolox.tools.train -n yolox-s-object-pose-ti-lite --dataset ycbv -c 'path to pretrained ckpt' -d 4 -b 64 --fp16 -o --task object_pose
                            -n yolox-m-object-pose-ti-lite           lmo              
                            -n yolox-l-object-pose-ti-lite           
```

## **Model Testing** 
The eval script computes ADD(s) score for a model. Apart from that, it generates a CSV file **'bop_test.csv'**. This file can be used to generate **AR,  AR<sub>VSD</sub>, AR<sub>MSSD</sub>, AR<sub>MSPD</sub>** using [bop_toolkit](https://github.com/thodan/bop_toolkit) repo.

Run the following command to compute ADD(s) metric on a pretrained checkpoint:
  ```
  python -m yolox.tools.eval -n yolox-s-object-pose           --dataset ycbv  -b 64 -d 4 -c "path to ckpt" --task object_pose --fp16 --fuse
                                yolox-s-object-pose-ti-lite 
  ```
Now, use the csv file generated by the above scirpt to get all the BOP metrics. Clone the [bop_toolkit](https://github.com/thodan/bop_toolkit) repo and install the rquired dependencies. Final evaluation command is:
  ```
  python eval_bop19_pose.py "path to the csv file"
  ```
<br/> 

###  **ONNX Export Including Detection and 6D Pose Estimation:** 
* Run the following command to export the entire model including the detection and 6D pose estimation part,
    ``` 
    python tools/export_onnx.py -n yolox-s-object-pose          -c "path to ckpt"  --output-name yolox_s_object_pose.onnx         --export-det --dataset ycbv --task object_pose
                                   yolox-s-object-pose-ti-lite                                   yolox_s_object_pose_ti_lite.onnx
    ```
* Apart from exporting the complete ONNX model, above script will generate a prototxt file that contains information of the detection layer. This prototxt file is required to deploy and acclerate the moodel on TI SoC.

###  **ONNXRT Inference: 6D Object Pose Estimation Inference with an End-to-End ONNX Model:**
 * If you haven't exported a model with the above command, download a sample model from here.
    * **Default Models**
        [yolox_s_object_pose.onnx](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/ycbv/edgeai-yolox/checkpoints/yolox_s_object_pose/yolox_s_object_pose.onnx),  [yolox_m_object_pose.onnx](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/ycbv/edgeai-yolox/checkpoints/yolox_m_object_pose/yolox_m_object_pose.onnx),  [yolox_l_object_pose.onnx](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/ycbv/edgeai-yolox/checkpoints/yolox_l_object_pose/yolox_l_object_pose.onnx)
    * **Lite Models and prototxt**
      * **Models**: [yolox_s_object_pose_ti_lite.onnx](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/ycbv/edgeai-yolox/checkpoints/yolox_s_object_pose_ti_lite/yolox_s_object_pose_ti_lite.onnx),  [yolox_m_object_pose_ti_lite.onnx](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/ycbv/edgeai-yolox/checkpoints/yolox_m_object_pose_ti_lite/yolox_m_object_pose_ti_lite.onnx),  [yolox_l_object_pose_ti_lite.onnx](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/ycbv/edgeai-yolox/checkpoints/yolox_l_object_pose_ti_lite/yolox_l_object_pose_ti_lite.onnx)
      * **Prototxt** : Prototxt files are required to run the lite models efficiently on TI processors.
        [yolox_s_object_pose_ti_lite.prototxt](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/ycbv/edgeai-yolox/checkpoints/yolox_s_object_pose_ti_lite/yolox_s_object_pose_ti_lite.prototxt),  [yolox_m_object_pose_ti_lite.prototxt](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/ycbv/edgeai-yolox/checkpoints/yolox_m_object_pose_ti_lite/yolox_m_object_pose_ti_lite.prototxt),  [yolox_l_object_pose_ti_lite.prototxt](http://software-dl.ti.com/jacinto7/esd/modelzoo/08_05_00_01/models/vision/object_6d_pose/ycbv/edgeai-yolox/checkpoints/yolox_l_object_pose_ti_lite/yolox_l_object_pose_ti_lite.prototxt)

 * Run the script as below to run inference with an ONNX model. The script runs inference and visualize the results. 
    ``` 
    cd demo/ONNXRuntime
    # Run inference on a set of sample images
    python onnx_inference_object_pose.py --model "path_to_onnx_model"  --image-folder "path-to-input-images" --output-dir "ops_onnxrt" 
    ```
 * The ONNX model infers the six dimenaional rerpresentation of the rotation parameter. The rotation matrix is deterministically computed from this 6D representation. This is done outside the model.
 * The camera parameters for a given dataset are part of the ONNX model. The exact translation parameters are computed from it's 2D proejection. This is computed inside the ONNX model.

###  **ONNXRT Inference on TDA4X: 6D Object Pose Estimation Import and Inference with an End-to-End ONNX Model:**
* Similar to running the model on PC using ONNXRT, it can be deployed in TI SOC in a similar fashion. Select the ti-lite models here as they are specifically optimized for TI processors.
* You can follow the installation instruction from [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools). A new environemnet must be created before thid setup. Make sure yor installation is correct by running some [python examples](https://github.com/TexasInstruments/edgeai-tidl-tools#python-examples).
* Here, we need to compile the model first as shown below. This will convert the floating point model to fix-point. This is done in PC.
    ``` 
    python onnx_inference_object_pose.py ---compile --model "path_to_onnx_model" --image-folder "path-to-input-images" --dst-path "ops_onnxrt_ti"   #Is dst-path needed here?
    ```
* After compilation, the model can be deployed on TI SOC for inference. For sanity check, one can run inference on PC as well. 
    ``` 
    # Run inference on a set of sample images as specified by image-folder
    python onnx_inference_object_pose.py ---inference --model "path_to_onnx_model" --image-folder "path-to-input-images" --dst-path "ops_onnxrt_ti"  
    ```
