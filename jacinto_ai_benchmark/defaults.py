import os
import cv2
from . import preprocess, postprocess, constants


###############################################################
# some utility functions to easily create the pipeline
def get_preproc_vgg(resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False, backend='pil',
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


def get_preproc_inception(resize=256, crop=224, data_layout=constants.NHWC, reverse_channels=False, backend='pil',
                          mean=(128.0, 128.0, 128.0), scale=(1/128.0, 1/128.0, 1/128.0)):
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


def get_postproc_detection(save_detections=True):
    postprocess_detection = [postprocess.Concat(axis=-1, end_index=3),
                             postprocess.IndexArray(),
                             postprocess.DetectionFilter(),
                             postprocess.DetectionResize()]
    if save_detections:
        postprocess_detection += [postprocess.DetectionImageSave()]
    #
    return postprocess_detection


############################################################
# quantization / calibration params
class QuantizationParams():
    def __init__(self, tidl_tensor_bits, max_frames_calib, max_calib_iterations, is_qat=False):
        self.tidl_tensor_bits = tidl_tensor_bits
        self.max_frames_calib = max_frames_calib
        self.max_calib_iterations = max_calib_iterations
        self.is_qat = is_qat

    def get_num_frames_calib(self):
        return self.max_frames_calib if self.tidl_tensor_bits == 8 else 1

    def get_num_calib_iterations(self):
        return self.max_calib_iterations if self.tidl_tensor_bits == 8 else 1

    def get_calib_options_tvm(self):
        calib_options_tvm_dict = {
            8:  {
                    'activation_range': 'on',
                    'weight_range': 'on',
                    'bias_calibration': 'on',
                    'per_channel_weight': 'off',
                    'iterations': self.get_num_calib_iterations(),
                 },
            16: {
                    'activation_range': 'off',
                    'weight_range': 'off',
                    'bias_calibration': 'off',
                    'per_channel_weight': 'off',
                    'iterations': self.get_num_calib_iterations(),
                 },
            32: {
                    'activation_range': 'off',
                    'weight_range': 'off',
                    'bias_calibration': 'off',
                    'per_channel_weight': 'off',
                    'iterations': self.get_num_calib_iterations(),
                 },
            'qat': {
                    'activation_range': 'off',
                    'weight_range': 'off',
                    'bias_calibration': 'off',
                    'per_channel_weight': 'off',
                    'iterations': self.get_num_calib_iterations(),
                 }
        }
        calib_opt = (calib_options_tvm_dict['qat'] if self.is_qat else \
                         calib_options_tvm_dict[self.tidl_tensor_bits])
        return calib_opt

    def get_session_tvm_dlr_cfg(self):
        session_tvm_dlr_cfg = {
            'tidl_tensor_bits': self.tidl_tensor_bits,
            'tidl_calibration_options': self.get_calib_options_tvm()
        }
        return session_tvm_dlr_cfg

    def get_calib_options_tflite_rt(self):
        calib_options_tflite_rt_dict = {
            8: {
                    "tidl_calibration_options:num_frames_calibration": self.get_num_frames_calib(),
                    "tidl_calibration_options:bias_calibration_iterations": self.get_num_calib_iterations()
            },
            16: {
                    "tidl_calibration_options:num_frames_calibration": self.get_num_frames_calib(),
                    "tidl_calibration_options:bias_calibration_iterations": self.get_num_calib_iterations()
            },
            32: {
                    "tidl_calibration_options:num_frames_calibration": self.get_num_frames_calib(),
                    "tidl_calibration_options:bias_calibration_iterations": self.get_num_calib_iterations()
            },
            'qat': {
                    "tidl_calibration_options:num_frames_calibration": self.get_num_frames_calib(),
                    "tidl_calibration_options:bias_calibration_iterations": self.get_num_calib_iterations()
            }
        }
        calib_opt = (calib_options_tflite_rt_dict['qat'] if self.is_qat else \
                         calib_options_tflite_rt_dict[self.tidl_tensor_bits])
        return calib_opt

    def get_session_tflite_rt_cfg(self):
        session_tflite_rt_cfg = {
            'tidl_tensor_bits': self.tidl_tensor_bits,
            'tidl_calibration_options': self.get_calib_options_tflite_rt()
        }
        return session_tflite_rt_cfg