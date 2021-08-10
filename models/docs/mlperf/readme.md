# MLPerf Machine Learning Benchmark Suite - Benchmark Results

MLPerf is a Machine Learning Benchmark that provides various kinds of Machine Learning Benchmarks.
Quanting from their website [mlperf.org](mlperf.org), the benchmark aims to do: "Fair and useful benchmarks for measuring training and inference performance of ML hardware, software, and services". 

The software and models for the benchmarks are provided in their github repositories in [https://github.com/mlcommons](https://github.com/mlcommons) and [https://github.com/mlperf](https://github.com/mlperf). Benchmarks are provides for both training and inference of models. Inference include Cloud scenario as well as Edge scenario. 

In this repository, we focus on inference of models for edge and mobile benchmarks. 


## Models used in the benchmark
- Documentation and Models used in MLPerf Edge benchmark: [link](https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection) 
- Documentation and Models used in MLPerf Mobile benchmark: [link](https://github.com/mlcommons/mobile_models)


## Model preparation
Some of the models have to processed with additional steps before they can be used in TIDL. These steps are described here:<br> 
[**MLPerf model preparation for TIDL**](./mlperf_models.md)


## Models
- [MLPerf Classification Models Link](../../vision/classification/imagenet1k/mlperf/)
- [MLPerf Detection Models Link](../../vision/detection/coco/mlperf/)
- [MLPerf Segmentation Models Link](../../vision/segmentation/ade20k32/mlperf/)

|Dataset         |Model Name         |Input Size|GigaMACs  |Accuracy%         |Task          |Available|Notes     |
|---------       |----------         |----------|----------|--------         |-------------- |---------|----------|
|ImageNet        |MobileNetV1        |224x224   |0.569     |71.676 Top-1%    |Classification |Y        |mobilenet_v1_1.0_224.tflite|
|ImageNet        |ResNet50           |224x224   |4.096     |76.456 Top-1%    |Classification |Y        |resnet50_v1.5.tflite|
|ImageNet        |MobileNetV2EdgeTPU |224x224   |0.991     |75.6   Top-1%    |Classification |Y        |mobilenet_edgetpu_224_1.0_float.tflite|
|COCO            |MobileNetV1SSD     |300x300   |1.237     |23.0 AP[.5:.95]% |Detection      |Y        |ssd_mobilenet_v1_coco_2018_01_28.tflite|
|COCO            |MobileNetV2SSD     |300x300   |1.875     |22.0 AP[.5:.95]% |Detection      |Y        |ssd_mobilenet_v2_300_float.tflite|
|COCO            |ResNet34SSD        |1200x1200 |          |20.0 AP[.5:.95]% |Detection      |Y        |ssd_resnet34-ssd1200.onnx|
|ADE20K 32 Class |DeepLabV3LiteMNV2  |512x512   |5.336     |54.8 MeanIoU%    |Segmentation   |Y        |deeplabv3_mnv2_ade20k_float.tflite|


Notes: 
- The attribution to their original authors and the licenses of the original sources are in a license.txt file along with each model.
- GigaMACS: Complexity in Giga Multipy-Accumulations (lower, the better).
- Accuracy: Accuracy obtained after the training, as reported in the MLPerf documentation.


## References
[1] MLPerf Inference Benchmark, Vijay Janapa Reddi and Christine Cheng and David Kanter and Peter Mattson and Guenther Schmuelling and Carole-Jean Wu and Brian Anderson and Maximilien Breughe and Mark Charlebois and William Chou and Ramesh Chukka and Cody Coleman and Sam Davis and Pan Deng and Greg Diamos and Jared Duke and Dave Fick and J. Scott Gardner and Itay Hubara and Sachin Idgunji and Thomas B. Jablin and Jeff Jiao and Tom St. John and Pankaj Kanwar and David Lee and Jeffery Liao and Anton Lokhmotov and Francisco Massa and Peng Meng and Paulius Micikevicius and Colin Osborne and Gennady Pekhimenko and Arun Tejusve Raghunath Rajan and Dilip Sequeira and Ashish Sirasao and Fei Sun and Hanlin Tang and Michael Thomson and Frank Wei and Ephrem Wu and Lingjie Xu and Koichi Yamada and Bing Yu and George Yuan and Aaron Zhong and Peizhao Zhang and Yuchen Zhou, 2019, arXiv:1911.02549, https://arxiv.org/abs/1911.02549

[2] MlCommons: Machine learning innovation to benefit everyone, https://mlcommons.org/en/

[3] MLPerf Inference Benchmark Suite: https://github.com/mlcommons/inference 

[4] MLPerf Inference Overview: https://mlperf.org/inference-overview

[5] MLPerf Inference v0.7 Results: https://mlperf.org/inference-results-0-7
