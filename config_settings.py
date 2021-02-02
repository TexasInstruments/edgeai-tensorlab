from jacinto_ai_benchmark import preprocess, postprocess, constants
from jacinto_ai_benchmark.defaults import *

############################################################
# common settings
num_frames = 10000 #50000
num_frames_calibration = 100 #10 #100
bias_calibration_iterations = 50 #10 #50
modelzoo_path = '../jacinto-ai-modelzoo/models'
datasets_path = f'./dependencies/datasets'
cuda_devices = [0,1,2,3,0,1,2,3] #None
tidl_dir = './dependencies/c7x-mma-tidl'
tidl_tensor_bits = 32 #8 #32

############################################################
# dataset settings
imagenet_train_cfg = dict(
    path=f'{datasets_path}/imagenet/train',
    split=f'{datasets_path}/imagenet/train.txt',
    shuffle=True,num_frames=num_frames_calibration)
imagenet_val_cfg = dict(
    path=f'{datasets_path}/imagenet/val',
    split=f'{datasets_path}/imagenet/val.txt',
    shuffle=True,num_frames=num_frames)

############################################################
# tvm_dlr session settings
tidl_calibration_options_tvm_dict = {
    8 :  {
            'activation_range' : 'on',
            'weight_range' : 'on',
            'bias_calibration' : 'on',
            'per_channel_weight' : 'off',
            'iterations' : bias_calibration_iterations,
         },
    16 : {
            'activation_range' : 'off',
            'weight_range' : 'off',
            'bias_calibration' : 'off',
            'per_channel_weight' : 'off',
            'iterations' : 1,
         },
    32 : {
            'activation_range' : 'off',
            'weight_range' : 'off',
            'bias_calibration' : 'off',
            'per_channel_weight' : 'off',
            'iterations' : 1,
         },
    'qat' : {
            'activation_range' : 'off',
            'weight_range' : 'off',
            'bias_calibration' : 'off',
            'per_channel_weight' : 'off',
            'iterations' : 1,
         }
}

tidl_calibration_options_tvm=tidl_calibration_options_tvm_dict[tidl_tensor_bits]
tidl_calibration_options_tvm_qat=tidl_calibration_options_tvm_dict['qat']

session_tvm_dlr_cfg = {
    'tidl_tensor_bits':tidl_tensor_bits,
    'tidl_calibration_options':tidl_calibration_options_tvm
}
session_tvm_dlr_cfg_qat = {
    'tidl_tensor_bits':tidl_tensor_bits,
    'tidl_calibration_options':tidl_calibration_options_tvm_qat
}

############################################################
# tflite_rt session settings

tidl_calibration_options_tflite_rt_dict = {
    8 : {
            "tidl_calibration_options:num_frames_calibration":num_frames_calibration,
             "tidl_calibration_options:bias_calibration_iterations":bias_calibration_iterations
    },
    16 : {
            "tidl_calibration_options:num_frames_calibration":1,
             "tidl_calibration_options:bias_calibration_iterations":1
    },
    32 : {
            "tidl_calibration_options:num_frames_calibration":1,
             "tidl_calibration_options:bias_calibration_iterations":1
    },
    'qat' : {
            "tidl_calibration_options:num_frames_calibration":1,
             "tidl_calibration_options:bias_calibration_iterations":1
    }
}
tidl_calibration_options_tflite_rt=tidl_calibration_options_tflite_rt_dict[tidl_tensor_bits]
tidl_calibration_options_tflite_rt_qat = tidl_calibration_options_tflite_rt_dict['qat']

session_tflite_rt_cfg = {
    'tidl_tensor_bits':tidl_tensor_bits,
    'tidl_calibration_options':tidl_calibration_options_tflite_rt
}
session_tflite_rt_cfg_qat = {
    'tidl_tensor_bits':tidl_tensor_bits,
    'tidl_calibration_options':tidl_calibration_options_tflite_rt_qat
}

