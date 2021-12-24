# EdgeAI-MMDetection Usage


## Scripts
Additional scripts are provided on top of mmdetection to ease the training and testing process. Several complexity optimized configurations are provided in the folder [configs/edgeailite](../configs/edgeailite). Scripts for training or evaluation with these configs are given in the [scripts](../scripts) folder. They can be executed by running the shell scripts provided. 


## Training
Select the appropriate config file in [detection_configs.py](../scripts/detection_configs.py)

Start the training by running [run_detection_train.sh](../run_detection_train.sh) 

After doing the floating point training, it is possible to run Qunatization Aware Training (QAT) starting from the trained checkpoint. For this, set quantize = True in the config file (see the line where it is set to False and change it to True) and run the training again. This will run a small number epochs of fine tuning with QAT at a lower learning rate.


## Evaluation/Testing
Make sure that the appropriate config file is selected in [detection_configs.py](../scripts/detection_configs.py)

Start evaluation by running [run_detection_test.sh](../run_detection_test.sh).

Note: If you did QAT, then the flag quantize in teh config file must be set to True even at this stage. 


## ONNX & Prototxt Export
Make sure that the appropriate config file is selected in [detection_configs.py](../scripts/detection_configs.py)

Start export by running [run_detection_export.sh](../run_detection_export.sh).

Note: If you did QAT, then the flag quantize in the config file must be set to True even at this stage. 


## Embedded friendly models
We now have a tool that can automatically replace some of embedded un-friendly layers. This tool will run automatically if you set the parameter convert_to_lite_model in the config file. Please see our edgeailite config files for some example usage. Examples: 
- convert_to_lite_model = dict(group_size_dw=1)
- convert_to_lite_model = dict(group_size_dw=None)
- where group_size_dw indicates the (convolution) group size to be used for convolution layers with kernel size larger than 1.
