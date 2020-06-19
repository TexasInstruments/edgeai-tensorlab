# Jacinto-AI-MMDetection Usage

Additional scripts are provided on top of mmdetection to ease the training and testing process. 

#### Training
- Several complexity optimized configurations are provided in the folder pytorch-mmdetection/configs/jacinto_ai
- Training is done by pytorch-mmdetection/scripts/train_main.py or pytorch-mmdetection/scripts/train_dist.py (Select the appropriate config file inside these scripts).
- To enable quantization during training, the quantize flag in the config file being used must be a "truth value in Python" - i.e. a string or True or something like that. If quantize is commented out or if it is False, None etc, quantization will not be performed.
- Once the training is done test can be done by pytorch-mmdetection/scripts/test_main.py or pytorch-mmdetection/scripts/test_dist.py (Select the appropriate config file inside these scripts).

## Testing
- Test can be done by using the scripts ./scripts/test_main.py or ./scripts/test_dist.py
- To enable quantization during test, the quantize flag in the config file being used must be a "truth value in Python" - i.e. a string or True or something like that. If quantize is commented out or if it is False, None etc, quantization will not be performed.