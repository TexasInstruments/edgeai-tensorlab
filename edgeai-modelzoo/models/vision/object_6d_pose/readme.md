# YOLO-6D-Pose Multi-Object 6D Pose Estimation Model

## Introduction
6D pose estimation is the task of estimating he 3D orientation and 3D translation of objects in a given environment. It is useful in a wide range of applications like robotic manipulation for bin picking, motion planning, and human-robot interaction task such as learning from demonstration.

### YOLO-6D-Pose

It is a novel single stage, end-to-end approach based on YOLOX that takes a single image as an input and infer accurate pose of each object  without any intermediate representation or complex post-processing like refinement. The model use a parameterized version of the 6D pose from which the rotation and translation parameters are recovered with a simple transformation.

These models are completely accelerated in TIDL. For more details about the architecture and training details, refer to these repository [YOLOX](https://github.com/TexasInstruments/edgeai-yolox).


|Dataset |          Model Name            |Input Size |GFLOPS| AR  | AR<sub>VSD</sub>| AR<sub>MSSD</sub>|AR<sub>MSPD</sub>|ADD(s)| Notes |
|--------|------------------------------- |-----------|------|-----|-----------------|------------------|-----------------|------|-------|
|YCBV    | YOLOX_s_object_pose_ti_lite    |640x480    | 31.2 |64.73|      59.53      |      65.81       |     68.86       | 54.12 |[pretrained_weights](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth)|


## References
[1] Optimized YOLOX based models from Texas Instruments, https://github.com/TexasInstruments/edgeai-yolox

