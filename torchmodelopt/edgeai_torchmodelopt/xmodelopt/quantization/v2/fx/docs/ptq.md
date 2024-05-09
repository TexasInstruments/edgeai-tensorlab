## Post Training Quantization in TIDL (PTQ)
Please consult the TIDL documentation to understand the options to be used for getting the best accuracy with PTQ. If you are using Open Source Runtimes (OSRT) of TIDL, then [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) has the documentation and examples. Additional examples are provided in [edgeai-benchmark](https://github.com/TexasInstruments/edgeai-benchmark), which provides model compilation options for the models in [edgeai-modelzoo](https://github.com/TexasInstruments/edgeai-modelzoo). 

Certain models such that has regression (continuous) outputs may need special handling to get the best accuracy with PTQ. Examples for such models are Object Detection models and Depth Estimation models. It is seen that Mixed Precision is a good way to improve the accuracy of such models. Mixed Precision here means using 16 bits for a some selected layers. It is seen that it is especially beneficial to put the first and last convolutional layers into 16 bits. 16bit layers can be easily specified by 'advanced_options:output_feature_16bit_names_list' in TIDL's OSRT options. Please see the examples [here](https://github.com/TexasInstruments/edgeai-benchmark/blob/master/configs/depth_estimation.py) and [here](https://github.com/TexasInstruments/edgeai-benchmark/blob/master/configs/detection.py)


### PTQ Compilation options for TIDL

#### Instructions for PTQ compiling models in TIDL
If you are using TIDL to infer a model trained using QAT tools provided in this repository, please set the following in the import config file of TIDL for best accuracy with PTQ: <br>
```
calibrationOption = 7  #advanced PTQ option
numFrames = 50
biasCalibrationIterations = 50
```


#### Instructions for PTQ compiling models in Open Source Runtimes of TIDL:
The compilation options to be used to get the best accuracy with PTQ in ONNXRuntime and TFLiteRuntime for TIDL are:
```
'accuracy_level': 1  #enable advanced PTQ
'advanced_options:calibration_frames': 50
'advanced_options:calibration_iterations': 50
'advanced_options:activation_clipping': 1
'advanced_options:weight_clipping': 1
'advanced_options:bias_calibration': 1
```

