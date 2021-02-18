from jacinto_ai_benchmark.defaults import *

############################################################
# execution pipeline type - currently only accuracy pipeline is defined
type = 'accuracy'
# number of frames for inference
num_frames = 10000 #50000
# number of frames to be used for post training quantization / calibration
max_frames_calib = 50 #100
# number of itrations to be used for post training quantization / calibration
max_calib_iterations = 50
# clone the modelzoo repo and make sure this folder is available.
modelzoo_path = '../jacinto-ai-modelzoo/models'
# create your datasets under this folder
datasets_path = f'./dependencies/datasets'
# important parameter. set this to 'pc' to do import and inference in pc
# set this to 'j7' to run inference in device. for inference on device run_import
# below should be switched off and it is assumed that the artifacts are already created.
target_device = 'pc' #'j7' #'pc'
# for parallel execution on cpu or gpu. if you don't have gpu, these actual numbers don't matter,
# but the size of teh list determines the number of parallel processes
# if you have gpu's these entries can be gpu ids which will be used to set CUDA_VISIBLE_DEVICES
parallel_devices = None #[0,1,2,3,0,1,2,3]
# quantization bit precision
tidl_tensor_bits = 8 #8 #16 #32
# run import of the model - only to be used in pc - set this to False for j7 evm
# for pc this can be True or False
run_import = True
# run inference - for inferene in j7 evm, it is assumed that the artifaacts folders are already available
run_inference = True
# collect final accuracy results
collect_results = True
# detection threshold
detection_thr = 0.05
# save detection, segmentation output
save_output = False
# wild card list to match against the model_path - only matching models will be run
# examples: ['classification'] ['imagenet1k'] ['torchvision']
# examples: ['resnet18_opset9.onnx', 'resnet50_v1.tflite']
model_selection = None
# verbose mode - print out more information
verbose = False


############################################################
# quantization params & session config
quantization_params = QuantizationParams(tidl_tensor_bits, max_frames_calib, max_calib_iterations)
session_tvm_dlr_cfg = quantization_params.get_session_tvm_dlr_cfg()
session_tflite_rt_cfg = quantization_params.get_session_tflite_rt_cfg()


quantization_params_qat = QuantizationParams(tidl_tensor_bits, max_frames_calib, max_calib_iterations, is_qat=True)
session_tvm_dlr_cfg_qat = quantization_params_qat.get_session_tvm_dlr_cfg()
session_tflite_rt_cfg_qat = quantization_params_qat.get_session_tflite_rt_cfg()


############################################################
# dataset settings
imagenet_cls_calib_cfg = dict(
    path=f'{datasets_path}/imagenet/val',
    split=f'{datasets_path}/imagenet/val.txt',
    shuffle=True,
    num_frames=quantization_params.get_num_frames_calib())

imagenet_cls_val_cfg = dict(
    path=f'{datasets_path}/imagenet/val',
    split=f'{datasets_path}/imagenet/val.txt',
    shuffle=True,
    num_frames=min(num_frames,50000))


coco_det_calib_cfg = dict(
    path=f'{datasets_path}/coco',
    split='val2017',
    shuffle=True,
    num_frames=quantization_params.get_num_frames_calib())

coco_det_val_cfg = dict(
    path=f'{datasets_path}/coco',
    split='val2017',
    shuffle=False, #TODO: need to make COCODetection.evaluate() work with shuffle
    num_frames=min(num_frames,5000))


cityscapes_seg_calib_cfg = dict(
    path=f'{datasets_path}/cityscapes',
    split='val',
    shuffle=True,
    num_frames=quantization_params.get_num_frames_calib())

cityscapes_seg_val_cfg = dict(
    path=f'{datasets_path}/cityscapes',
    split='val',
    shuffle=True,
    num_frames=min(num_frames,500))


ade20k_seg_calib_cfg = dict(
    path=f'{datasets_path}/ADEChallengeData2016',
    split='validation',
    shuffle=True,
    num_frames=quantization_params.get_num_frames_calib())

ade20k_seg_val_cfg = dict(
    path=f'{datasets_path}/ADEChallengeData2016',
    split='validation',
    shuffle=True,
    num_frames=min(num_frames, 2000))



