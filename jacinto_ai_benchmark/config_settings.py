import cv2
from . import utils, preprocess, postprocess, constants
from . import config_dict


class ConfigSettings(config_dict.ConfigDict):
    def __init__(self, input, **kwargs):
        super().__init__(input, **kwargs)

        ############################################################
        # quantization params & session config
        ###############################################################
        self.quantization_params = QuantizationParams(self.tidl_tensor_bits, self.max_frames_calib, self.max_calib_iterations)
        self.session_tvm_dlr_cfg = self.quantization_params.get_session_tvm_dlr_cfg()
        self.session_tflite_rt_cfg = self.quantization_params.get_session_tflite_rt_cfg()

        self.quantization_params_qat = QuantizationParams(self.tidl_tensor_bits, self.max_frames_calib, self.max_calib_iterations, is_qat=True)
        self.session_tvm_dlr_cfg_qat = self.quantization_params_qat.get_session_tvm_dlr_cfg()
        self.session_tflite_rt_cfg_qat = self.quantization_params_qat.get_session_tflite_rt_cfg()

        ############################################################
        # dataset settings
        ###############################################################
        self.imagenet_cls_calib_cfg = dict(
            path=f'{self.datasets_path}/imagenet/val',
            split=f'{self.datasets_path}/imagenet/val.txt',
            shuffle=True,
            num_frames=self.quantization_params.get_num_frames_calib())

        self.imagenet_cls_val_cfg = dict(
            path=f'{self.datasets_path}/imagenet/val',
            split=f'{self.datasets_path}/imagenet/val.txt',
            shuffle=True,
            num_frames=min(self.num_frames,50000))

        self.coco_det_calib_cfg = dict(
            path=f'{self.datasets_path}/coco',
            split='val2017',
            shuffle=True,
            num_frames=self.quantization_params.get_num_frames_calib())

        self.coco_det_val_cfg = dict(
            path=f'{self.datasets_path}/coco',
            split='val2017',
            shuffle=False, #TODO: need to make COCODetection.evaluate() work with shuffle
            num_frames=min(self.num_frames,5000))

        self.cityscapes_seg_calib_cfg = dict(
            path=f'{self.datasets_path}/cityscapes',
            split='val',
            shuffle=True,
            num_frames=self.quantization_params.get_num_frames_calib())

        self.cityscapes_seg_val_cfg = dict(
            path=f'{self.datasets_path}/cityscapes',
            split='val',
            shuffle=True,
            num_frames=min(self.num_frames,500))

        self.ade20k_seg_calib_cfg = dict(
            path=f'{self.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=self.quantization_params.get_num_frames_calib())

        self.ade20k_seg_val_cfg = dict(
            path=f'{self.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=min(self.num_frames, 2000))

        self.voc_seg_calib_cfg = dict(
            path=f'{self.datasets_path}/VOCdevkit/VOC2012',
            split='val',
            shuffle=True,
            num_frames=self.quantization_params.get_num_frames_calib())

        self.voc_seg_val_cfg = dict(
            path=f'{self.datasets_path}/VOCdevkit/VOC2012',
            split='val',
            shuffle=True,
            num_frames=min(self.num_frames, 1449))


    ###############################################################
    # preprocess transforms
    ###############################################################
    def _get_preproc_base(self, resize, crop, data_layout, reverse_channels,
                         backend, interpolation, resize_with_pad, mean, scale):
        transforms_list = [
            preprocess.ImageRead(backend=backend),
            preprocess.ImageResize(resize, interpolation=interpolation, resize_with_pad=resize_with_pad),
            preprocess.ImageCenterCrop(crop),
            preprocess.ImageToNPTensor4D(data_layout=data_layout),
            preprocess.ImageNormMeanScale(mean=mean, scale=scale, data_layout=data_layout)]
        if reverse_channels:
            transforms_list = transforms_list + [preprocess.NPTensor4DChanReverse(data_layout=data_layout)]
        #
        transforms = utils.TransformsCompose(transforms_list, resize=resize, crop=crop,
                                             data_layout=data_layout, reverse_channels=reverse_channels,
                                             backend=backend, interpolation=interpolation,
                                             mean=mean, scale=scale)
        return transforms

    def get_preproc_onnx(self, resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
                         backend='pil', interpolation=None, resize_with_pad=False,
                         mean=(123.675, 116.28, 103.53), scale=(0.017125, 0.017507, 0.017429)):
        transforms = self._get_preproc_base(resize=resize, crop=crop, data_layout=data_layout,
                                      reverse_channels=reverse_channels, backend=backend, interpolation=interpolation,
                                      resize_with_pad=resize_with_pad, mean=mean, scale=scale)
        return transforms

    def get_preproc_jai(self, resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False,
                        mean=(128.0, 128.0, 128.0), scale=(1/64.0, 1/64.0, 1/64.0)):
        return self._get_preproc_base(resize=resize, crop=crop, data_layout=data_layout, reverse_channels=reverse_channels,
                                backend=backend, interpolation=interpolation, resize_with_pad=resize_with_pad,
                                mean=mean, scale=scale)

    def get_preproc_mxnet(self, resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=None, resize_with_pad=False,
                        mean=(123.675, 116.28, 103.53), scale=(0.017125, 0.017507, 0.017429)):
        return self._get_preproc_base(resize=resize, crop=crop, data_layout=data_layout, reverse_channels=reverse_channels,
                                backend=backend, interpolation=interpolation, resize_with_pad=resize_with_pad,
                                mean=mean, scale=scale)

    def get_preproc_tflite(self, resize=256, crop=224, data_layout=constants.NHWC, reverse_channels=False,
                              backend='pil', interpolation=None, resize_with_pad=False,
                              mean=(128.0, 128.0, 128.0), scale=(1/128.0, 1/128.0, 1/128.0)):
        return self._get_preproc_base(resize=resize, crop=crop, data_layout=data_layout, reverse_channels=reverse_channels,
                                backend=backend, interpolation=interpolation, resize_with_pad=resize_with_pad,
                                mean=mean, scale=scale)

    def get_postproc_classification(self):
        postprocess_classification = [postprocess.IndexArray(), postprocess.ArgMax()]
        transforms = utils.TransformsCompose(postprocess_classification)
        return transforms

    ###############################################################
    # post process transforms for detection
    ###############################################################
    def _get_postproc_detection_base(self, formatter=None, resize_with_pad=False,
                               normalized_detections=True, shuffle_indices=None):
        postprocess_detection = [postprocess.ShuffleList(indices=shuffle_indices),
                                 postprocess.Concat(axis=-1, end_index=3),
                                 postprocess.IndexArray()]
        if formatter is not None:
            postprocess_detection += [formatter]
        #
        postprocess_detection += [postprocess.DetectionResizePad(resize_with_pad=resize_with_pad,
                                                    normalized_detections=normalized_detections)]
        if self.detection_thr is not None:
            postprocess_detection += [postprocess.DetectionFilter(detection_thr=self.detection_thr,
                                                                  detection_max=self.detection_max)]
        #
        if self.save_output:
            postprocess_detection += [postprocess.DetectionImageSave()]
        #
        transforms = utils.TransformsCompose(postprocess_detection, detection_thr=self.detection_thr,
                                             save_output=self.save_output, formatter=formatter)
        return transforms

    def get_postproc_detection_onnx(self, formatter=None):
        return self._get_postproc_detection_base(formatter=formatter)

    def get_postproc_detection_tflite(self, formatter=postprocess.DetectionYXYX2XYXY()):
        return self._get_postproc_detection_base(formatter=formatter)

    def get_postproc_detection_mxnet(self, formatter=None, resize_with_pad=False,
                                normalized_detections=False, shuffle_indices=(2,0,1)):
        return self._get_postproc_detection_base(formatter=formatter, resize_with_pad=resize_with_pad,
                        normalized_detections=normalized_detections, shuffle_indices=shuffle_indices)

    ###############################################################
    # post process transforms for segmentation
    ###############################################################
    def _get_postproc_segmentation_base(self, data_layout, with_argmax=True):
        channel_axis = -1 if data_layout == constants.NHWC else 1
        postprocess_segmentation = [postprocess.IndexArray()]
        if with_argmax:
            postprocess_segmentation += [postprocess.ArgMax(axis=channel_axis)]
        #
        postprocess_segmentation += [postprocess.NPTensorToImage(data_layout=data_layout),
                                     postprocess.SegmentationImageResize()]
        if self.save_output:
            postprocess_segmentation += [postprocess.SegmentationImageSave()]
        #
        transforms = utils.TransformsCompose(postprocess_segmentation, data_layout=data_layout,
                                             save_output=self.save_output, with_argmax=with_argmax)
        return transforms

    def get_postproc_segmentation_onnx(self, data_layout=constants.NCHW, with_argmax=True):
        return self._get_postproc_segmentation_base(data_layout=data_layout, with_argmax=with_argmax)

    def get_postproc_segmentation_tflite(self, data_layout=constants.NHWC, with_argmax=True):
        return self._get_postproc_segmentation_base(data_layout=data_layout, with_argmax=with_argmax)


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