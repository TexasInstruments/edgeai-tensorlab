from jacinto_ai_benchmark import preprocess, postprocess, constants
from jacinto_ai_benchmark.defaults import *

############################################################
# common settings
num_frames = 10000 #50000
max_frames_calib = 100
max_calib_iterations = 50
modelzoo_path = '../jacinto-ai-modelzoo/models'
datasets_path = f'./dependencies/datasets'
cuda_devices = [0,1,2,3,0,1,2,3] #None
tidl_dir = './dependencies/c7x-mma-tidl'
tidl_tensor_bits = 32 #8 #16 #32


session_tvm_dlr_cfg = {
    'tidl_tensor_bits': tidl_tensor_bits,
    'tidl_calibration_options': get_calib_options_tvm(tidl_tensor_bits, max_frames_calib, max_calib_iterations)
}
session_tvm_dlr_cfg_qat = {
    'tidl_tensor_bits': tidl_tensor_bits,
    'tidl_calibration_options': get_calib_options_tvm('qat', max_frames_calib, max_calib_iterations)
}


session_tflite_rt_cfg = {
    'tidl_tensor_bits': tidl_tensor_bits,
    'tidl_calibration_options': get_calib_options_tflite_rt(tidl_tensor_bits, max_frames_calib, max_calib_iterations)
}
session_tflite_rt_cfg_qat = {
    'tidl_tensor_bits': tidl_tensor_bits,
    'tidl_calibration_options': get_calib_options_tflite_rt('qat', max_frames_calib, max_calib_iterations)
}


############################################################
# dataset settings
imagenet_train_cfg = dict(
    path=f'{datasets_path}/imagenet/train',
    split=f'{datasets_path}/imagenet/train.txt',
    shuffle=True,num_frames=get_num_frames_calib(tidl_tensor_bits, max_frames_calib))
imagenet_val_cfg = dict(
    path=f'{datasets_path}/imagenet/val',
    split=f'{datasets_path}/imagenet/val.txt',
    shuffle=True,num_frames=num_frames)



