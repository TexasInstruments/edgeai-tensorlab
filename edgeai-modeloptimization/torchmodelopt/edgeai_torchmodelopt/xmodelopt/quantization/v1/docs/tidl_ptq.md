## Post Training Quantization in TIDL (PTQ)
Please consult the TIDL documentation to understand the options to be used for getting the best accuracy with PTQ. If you are using Open Source Runtimes (OSRT) of TIDL, then [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) has the documentation and examples. Additional examples are provided in [edgeai-benchmark](https://github.com/TexasInstruments/edgeai-benchmark), which provides model compilation options for the models in [edgeai-modelzoo](https://github.com/TexasInstruments/edgeai-modelzoo). 

Certain models such that has regression (continuous) outputs may need special handling to get the best accuracy with PTQ. Examples for such models are Object Detection models and Depth Estimation models. It is seen that Mixed Precision is a good way to improve the accuracy of such models. Mixed Precision here means using 16 bits for a some selected layers. It is seen that it is especially beneficial to put the first and last convolutional layers into 16 bits. 16bit layers can be easily specified by 'advanced_options:output_feature_16bit_names_list' in TIDL's OSRT options. Please see the examples [here](https://github.com/TexasInstruments/edgeai-benchmark/blob/master/configs/depth_estimation.py) and [here](https://github.com/TexasInstruments/edgeai-benchmark/blob/master/configs/detection.py)


### Guidelines For Training To Get Best Accuracy With PTQ

**These are important** - we are listing these guidelines upfront because it is important to follow these.

We recommend that the training uses **sufficient amount of regularization (weight decay) for all parameters**. Regularization / weight decay ensures that the weights, biases and other parameters (if any) are small and compact - this is good for quantization. These features are supported in most of the popular training framework.<br>

We have noticed that some training code bases do not use weight decay for biases. Some other code bases do not use weight decay for the parameters in Depthwise convolution layers. All these are bad strategies for quantization. These poor choices done (probably to get a 0.1% accuracy lift with floating point) will result in a huge degradation in fixed point - sometimes several percentage points. The weight decay factor should not be too small. We have used a weight decay factor of 1e-4 for training several networks and we highly recommend a similar value. Please do no use small values such as 1e-5.<br>

We also highly recommend to use **Batch Normalization immediately after every Convolution layer**. This helps the feature map to be properly regularized/normalized. If this is not done, there can be accuracy degradation with quantization. This especially true for Depthwise Convolution layers. However applying Batch Normalization to the Convolution layer that does the prediction (for example, the very last layer in segmentation/object detection network) may hurt accuracy and can be avoided.<br>

There are several model types out there including MobileNets, ResNets, DenseNets, EfficientNets(Lite), RegNetX [9] etc. that are popular in the embedded community. The models using Depthwise convolutions (such as MobileNets) are more difficult to quantize - expect higher accuracy drop with quantization when using such models. We would like to **recommend RegNetX [9] models** as the most embedded friendly as they balance accuracy for a given complexity and the ease of quantization due to their use of grouped convolutions with carefully selected group sizes.<br>

To get the best accuracy at the quantization stage, it is important that the model is trained carefully, following the guidelines (even during the floating point training). Having spent a lot of time solving quantization issues, we would like to highlight that following these guidelines are of at most importance. Otherwise, there is a high risk that the tools and techniques described here may not be completely effective in solving the accuracy drop due to quantization. To summarize, if you are getting poor accuracy with quantization, please check the following:<br>
- Weight decay is applied to all layers / parameters and that weight decay factor is good.<br>
- Ensure that the Convolution layers in the network have Batch Normalization layers immediately after that. The only exception allowed to this rule is for the very last Convolution layer in the network (for example the prediction layer in a segmentation network or detection network, where adding Batch normalization might hurt the floating point accuracy).<br>


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

