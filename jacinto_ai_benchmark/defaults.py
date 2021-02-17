import cv2
from . import utils, preprocess, postprocess, constants


###############################################################
# some utility functions to easily create the pipeline
def get_preproc_onnx(resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
                     backend='pil', interpolation=None,
                     mean=(123.675, 116.28, 103.53), scale=(0.017125, 0.017507, 0.017429)):
    preprocess_tvm_dlr = [
        preprocess.ImageRead(backend=backend),
        preprocess.ImageResize(resize, interpolation=interpolation),
        preprocess.ImageCenterCrop(crop),
        preprocess.ImageToNPTensor4D(data_layout=data_layout),
        preprocess.ImageNormMeanScale(mean=mean, scale=scale, data_layout=data_layout)]
    if reverse_channels:
        preprocess_tvm_dlr = preprocess_tvm_dlr + [preprocess.NPTensor4DChanReverse(data_layout=data_layout)]
    #
    transforms = utils.TransformsCompose(preprocess_tvm_dlr, resize=resize, crop=crop,
                                         data_layout=data_layout, reverse_channels=reverse_channels,
                                         backend=backend, interpolation=interpolation,
                                         mean=mean, scale=scale)
    return transforms


def get_preproc_jai(resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
                    backend='cv2', interpolation=cv2.INTER_AREA,
                    mean=(128.0, 128.0, 128.0), scale=(1/64.0, 1/64.0, 1/64.0)):
    return get_preproc_onnx(resize=resize, crop=crop, data_layout=data_layout, reverse_channels=reverse_channels,
                            backend=backend, interpolation=interpolation, mean=mean, scale=scale)


def get_preproc_tflite(resize=256, crop=224, data_layout=constants.NHWC, reverse_channels=False,
                          backend='pil', interpolation=None,
                          mean=(128.0, 128.0, 128.0), scale=(1/128.0, 1/128.0, 1/128.0)):
    preprocess_tflite_rt = [
        preprocess.ImageRead(backend=backend),
        preprocess.ImageResize(resize, interpolation=interpolation),
        preprocess.ImageCenterCrop(crop),
        preprocess.ImageToNPTensor4D(data_layout=data_layout),
        preprocess.ImageNormMeanScale(mean=mean, scale=scale, data_layout=data_layout)]
    if reverse_channels:
        preprocess_tflite_rt = preprocess_tflite_rt + [preprocess.NPTensor4DChanReverse(data_layout=data_layout)]
    #
    transforms = utils.TransformsCompose(preprocess_tflite_rt, resize=resize, crop=crop,
                                         data_layout=data_layout, reverse_channels=reverse_channels,
                                         backend=backend, interpolation=interpolation,
                                         mean=mean, scale=scale)
    return transforms


def get_postproc_classification():
    postprocess_classification = [postprocess.IndexArray(), postprocess.ArgMax()]
    transforms = utils.TransformsCompose(postprocess_classification)
    return transforms


############################################################
def get_postproc_detection(detection_thr=None, save_output=True, formatter=None):
    postprocess_detection = [postprocess.Concat(axis=-1, end_index=3),
                             postprocess.IndexArray()]
    if formatter is not None:
        postprocess_detection += [formatter]
    #
    postprocess_detection += [postprocess.DetectionResize()]
    if detection_thr is not None:
        postprocess_detection += [postprocess.DetectionFilter(detection_thr=detection_thr)]
    #
    if save_output:
        postprocess_detection += [postprocess.DetectionImageSave()]
    #
    transforms = utils.TransformsCompose(postprocess_detection, detection_thr=detection_thr,
                                         save_output=save_output, formatter=formatter)
    return transforms


def get_postproc_detection_onnx(detection_thr=None, save_output=True, formatter=None):
    return get_postproc_detection(detection_thr=detection_thr, save_output=save_output, formatter=formatter)


def get_postproc_detection_tflite(detection_thr=None, save_output=True, formatter=postprocess.DetectionYXYX2XYXY()):
    return get_postproc_detection(detection_thr=detection_thr, save_output=save_output, formatter=formatter)


############################################################
def get_postproc_segmentation(data_layout, save_output, with_argmax=True):
    channel_axis = -1 if data_layout == constants.NHWC else 1
    postprocess_segmentation = [postprocess.IndexArray()]
    if with_argmax:
        postprocess_segmentation += [postprocess.ArgMax(axis=channel_axis)]
    #
    postprocess_segmentation += [postprocess.NPTensorToImage(data_layout=data_layout),
                                 postprocess.SegmentationImageResize()]
    if save_output:
        postprocess_segmentation += [postprocess.SegmentationImageSave()]
    #
    transforms = utils.TransformsCompose(postprocess_segmentation, data_layout=data_layout,
                                         save_output=save_output, with_argmax=with_argmax)
    return transforms


def get_postproc_segmentation_onnx(data_layout=constants.NCHW, save_output=True, with_argmax=True):
    return get_postproc_segmentation(data_layout=data_layout, save_output=save_output, with_argmax=with_argmax)


def get_postproc_segmentation_tflite(data_layout=constants.NHWC, save_output=True, with_argmax=True):
    return get_postproc_segmentation(data_layout=data_layout, save_output=save_output, with_argmax=with_argmax)


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