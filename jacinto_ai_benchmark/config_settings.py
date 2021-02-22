import cv2
from . import utils, preprocess, postprocess, constants


class ConfigSettings(utils.ConfigDict):
    def __init__(self, input):
        # execution pipeline type - currently only accuracy pipeline is defined
        self.type = 'accuracy'
        # number of frames for inference
        self.num_frames = 10000 #50000
        # number of frames to be used for post training quantization / calibration
        self.max_frames_calib = 50 #100
        # number of itrations to be used for post training quantization / calibration
        self.max_calib_iterations = 50
        # clone the modelzoo repo and make sure this folder is available.
        self.modelzoo_path = '../jacinto-ai-modelzoo/models'
        # create your datasets under this folder
        self.datasets_path = f'./dependencies/datasets'
        # important parameter. set this to 'pc' to do import and inference in pc
        # set this to 'j7' to run inference in device. for inference on device run_import
        # below should be switched off and it is assumed that the artifacts are already created.
        self.target_device = 'pc' #'j7' #'pc'
        # for parallel execution on cpu or gpu. if you don't have gpu, these actual numbers don't matter,
        # but the size of teh list determines the number of parallel processes
        # if you have gpu's these entries can be gpu ids which will be used to set CUDA_VISIBLE_DEVICES
        self.parallel_devices = [0,1,2,3,0,1,2,3] #None
        # quantization bit precision
        self.tidl_tensor_bits = 8 #8 #16 #32
        # run import of the model - only to be used in pc - set this to False for j7 evm
        # for pc this can be True or False
        self.run_import = True
        # run inference - for inferene in j7 evm, it is assumed that the artifaacts folders are already available
        self.run_inference = True #True
        # collect final accuracy results
        self.collect_results = True #True
        # detection threshold
        self.detection_thr = 0.05
        # save detection, segmentation output
        self.save_output = False
        # wild card list to match against the model_path - only matching models will be run
        # examples: ['classification'] ['imagenet1k'] ['torchvision']
        # examples: ['resnet18_opset9.onnx', 'resnet50_v1.tflite']
        self.model_selection = None
        # verbose mode - print out more information
        self.verbose = False
        super().__init__(input)
        self.initialize()

    def initialize(self):
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

    ###############################################################
    # preprocess transforms
    ###############################################################
    def get_preproc_onnx(self, resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
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


    def get_preproc_jai(self, resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA,
                        mean=(128.0, 128.0, 128.0), scale=(1/64.0, 1/64.0, 1/64.0)):
        return self.get_preproc_onnx(resize=resize, crop=crop, data_layout=data_layout, reverse_channels=reverse_channels,
                                backend=backend, interpolation=interpolation, mean=mean, scale=scale)


    def get_preproc_tflite(self, resize=256, crop=224, data_layout=constants.NHWC, reverse_channels=False,
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

    def get_postproc_classification(self):
        postprocess_classification = [postprocess.IndexArray(), postprocess.ArgMax()]
        transforms = utils.TransformsCompose(postprocess_classification)
        return transforms


    ###############################################################
    # post process transforms for detection
    ###############################################################
    def get_postproc_detection(self, detection_thr=None, save_output=True, formatter=None):
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

    def get_postproc_detection_onnx(self, detection_thr=None, save_output=True, formatter=None):
        return ConfigSettings.get_postproc_detection(detection_thr=detection_thr, save_output=save_output, formatter=formatter)

    def get_postproc_detection_tflite(self, detection_thr=None, save_output=True, formatter=postprocess.DetectionYXYX2XYXY()):
        return ConfigSettings.get_postproc_detection(detection_thr=detection_thr, save_output=save_output, formatter=formatter)


    ###############################################################
    # post process transforms for segmentation
    ###############################################################
    def get_postproc_segmentation(self, data_layout, save_output, with_argmax=True):
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

    def get_postproc_segmentation_onnx(self, data_layout=constants.NCHW, save_output=True, with_argmax=True):
        return ConfigSettings.get_postproc_segmentation(data_layout=data_layout, save_output=save_output, with_argmax=with_argmax)

    def get_postproc_segmentation_tflite(self, data_layout=constants.NHWC, save_output=True, with_argmax=True):
        return ConfigSettings.get_postproc_segmentation(data_layout=data_layout, save_output=save_output, with_argmax=with_argmax)


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