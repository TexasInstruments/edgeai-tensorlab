from jacinto_ai_benchmark import preprocess, postprocess, constants
from jacinto_ai_benchmark.defaults import *

############################################################
# define a few commonly used defautls here

num_frames = 10 #1000 #10 #50000
num_frames_calibration = 10 #100 #10 #100
bias_calibration_iterations = 50 #10 #50
modelzoo_path = '../jacinto-ai-modelzoo/models'
datasets_path = f'./dependencies/datasets'
cuda_devices = None #[0,1,2,3]
tidl_dir = './dependencies/c7x-mma-tidl'
tidl_tensor_bits = 32 #8 #32

imagenet_train_cfg = dict(path=f'{datasets_path}/imagenet/train', split=f'{datasets_path}/imagenet/train.txt',
                          shuffle=True,num_frames=num_frames_calibration)
imagenet_val_cfg = dict(path=f'{datasets_path}/imagenet/val', split=f'{datasets_path}/imagenet/val.txt',
                        shuffle=True,num_frames=num_frames)

tidl_calibration_options_tvm={"iterations":bias_calibration_iterations}

tidl_calibration_options_tflite_rt={"tidl_calibration_options:num_frames_calibration":num_frames_calibration,
                                 "tidl_calibration_options:bias_calibration_iterations":bias_calibration_iterations}

session_tvm_dlr_cfg = dict(tidl_tensor_bits=tidl_tensor_bits,
                           tidl_calibration_options=tidl_calibration_options_tvm)

session_tflite_rt_cfg = dict(tidl_tensor_bits=tidl_tensor_bits,
                             tidl_calibration_options=tidl_calibration_options_tflite_rt)

