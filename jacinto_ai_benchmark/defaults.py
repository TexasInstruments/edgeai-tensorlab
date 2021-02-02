from . import preprocess, postprocess, constants


###############################################################
# some utility functions to easily create the pipeline
def get_preproc_tvm_dlr(resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False, backend='pil',
                        mean=(123.675, 116.28, 103.53), scale=(0.017125, 0.017507, 0.017429)):
    preprocess_tvm_dlr = [
        preprocess.ImageRead(backend=backend),
        preprocess.ImageResize(resize),
        preprocess.ImageCenterCrop(crop),
        preprocess.ImageToNPTensor4D(data_layout=data_layout),
        preprocess.ImageNormMeanScale(mean=mean, scale=scale, data_layout=data_layout)]
    if reverse_channels:
        preprocess_tvm_dlr = preprocess_tvm_dlr + [preprocess.NPTensor4DChanReverse(data_layout=data_layout)]
    #
    return preprocess_tvm_dlr


def get_preproc_tflite_rt(resize=256, crop=224, data_layout=constants.NHWC, reverse_channels=False, backend='pil',
                          mean=(127.0, 127.0, 127.0), scale=(1/128.0, 1/128.0, 1/128.0)):
    preprocess_tflite_rt = [
        preprocess.ImageRead(backend=backend),
        preprocess.ImageResize(resize),
        preprocess.ImageCenterCrop(crop),
        preprocess.ImageToNPTensor4D(data_layout=data_layout),
        preprocess.ImageNormMeanScale(mean=mean, scale=scale, data_layout=data_layout)]
    if reverse_channels:
        preprocess_tflite_rt = preprocess_tflite_rt + [preprocess.NPTensor4DChanReverse(data_layout=data_layout)]
    #
    return preprocess_tflite_rt


def get_postproc_classification():
    postprocess_classification = [postprocess.IndexArray(), postprocess.ArgMax()]
    return postprocess_classification


############################################################
# common calibration settings
def get_num_frames_calib(tidl_tensor_bits, max_frames_calib):
    return max_frames_calib if tidl_tensor_bits == 8 else 1


def get_num_calib_iterations(tidl_tensor_bits, max_calib_iterations):
    return max_calib_iterations if tidl_tensor_bits == 8 else 1


############################################################
# tvm_dlr calibration settings
def get_calib_options_tvm(tidl_tensor_bits, max_frames_calib, max_calib_iterations):
    calib_options_tvm_dict = {
        8:  {
                'activation_range': 'on',
                'weight_range': 'on',
                'bias_calibration': 'on',
                'per_channel_weight': 'off',
                'iterations': get_num_calib_iterations(tidl_tensor_bits, max_calib_iterations),
             },
        16: {
                'activation_range': 'off',
                'weight_range': 'off',
                'bias_calibration': 'off',
                'per_channel_weight': 'off',
                'iterations': get_num_calib_iterations(tidl_tensor_bits, max_calib_iterations),
             },
        32: {
                'activation_range': 'off',
                'weight_range': 'off',
                'bias_calibration': 'off',
                'per_channel_weight': 'off',
                'iterations': get_num_calib_iterations(tidl_tensor_bits, max_calib_iterations),
             },
        'qat': {
                'activation_range': 'off',
                'weight_range': 'off',
                'bias_calibration': 'off',
                'per_channel_weight': 'off',
                'iterations': get_num_calib_iterations(tidl_tensor_bits, max_calib_iterations),
             }
    }
    return calib_options_tvm_dict[tidl_tensor_bits]


############################################################
# tflite_rt calibration settings
def get_calib_options_tflite_rt(tidl_tensor_bits, max_frames_calib, max_calib_iterations):
    calib_options_tflite_rt_dict = {
        8: {
                "tidl_calibration_options:num_frames_calibration": get_num_frames_calib(tidl_tensor_bits, max_frames_calib),
                "tidl_calibration_options:bias_calibration_iterations": get_num_calib_iterations(tidl_tensor_bits, max_calib_iterations)
        },
        16: {
                "tidl_calibration_options:num_frames_calibration": get_num_frames_calib(tidl_tensor_bits, max_frames_calib),
                "tidl_calibration_options:bias_calibration_iterations": get_num_calib_iterations(tidl_tensor_bits, max_calib_iterations)
        },
        32: {
                "tidl_calibration_options:num_frames_calibration": get_num_frames_calib(tidl_tensor_bits, max_frames_calib),
                "tidl_calibration_options:bias_calibration_iterations": get_num_calib_iterations(tidl_tensor_bits, max_calib_iterations)
        },
        'qat': {
                "tidl_calibration_options:num_frames_calibration": get_num_frames_calib(tidl_tensor_bits, max_frames_calib),
                "tidl_calibration_options:bias_calibration_iterations": get_num_calib_iterations(tidl_tensor_bits, max_calib_iterations)
        }
    }
    return calib_options_tflite_rt_dict[tidl_tensor_bits]

