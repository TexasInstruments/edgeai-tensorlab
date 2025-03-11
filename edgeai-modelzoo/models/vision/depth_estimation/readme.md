# Depth Estimation Benchmark


## Introduction
Depth Estimation using monocular images is an emerging field of research and hos shown interesting progress.


## Datasets
- **NYUDepthV2**: Nathan Silberman, Pushmeet Kohli, Derek Hoiem, Rob Fergus, Indoor Segmentation and Support Inference from RGBD Images, ECCV 2012, https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html


## Models

### Fast Depth
- [Models Link - NYUDepthV2](./nyudepthv2/fast-depth/)
- [Additional information](http://fastdepth.mit.edu/)
- [Training Code](https://github.com/dwofk/fast-depth)
- Note: Depth estimation from monocular image input


|Dataset    |Model Name                     |Input Size |GigaMACs  |Delta1%        |Available|Notes |
|-----------|------------------------------ |-----------|----------|---------------|---------|------|
|NYUDepthV2 |Fast Depth                     |224x224    |0.3825    |77.1           |Y        |      | 


### MiDaS
- [Models Link - NYUDepthV2](./nyudepthv2/MiDaS/)
- [Training Code](https://github.com/isl-org/MiDaS)
- Note: Depth estimation from monocular image input


|Dataset    |Model Name                     |Input Size |GigaMACs  |Delta1%        |Available|Notes |
|-----------|------------------------------ |-----------|----------|---------------|---------|------|
|NYUDepthV2 |MiDaS-small (v2.1)             |256x256    |4.633     |86.67          |Y        |      | 



## References

[1] NYUDepthV2: Nathan Silberman, Pushmeet Kohli, Derek Hoiem, Rob Fergus, Indoor Segmentation and Support Inference from RGBD Images, ECCV 2012, https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

[2] Wofk, Diana and Ma, Fangchang and Yang, Tien-Ju and Karaman, Sertac and Sze, Vivienne, FastDepth: Fast Monocular Depth Estimation on Embedded Systems, IEEE International Conference on Robotics and Automation (ICRA), 2019

[3] Rene Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun, Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020
