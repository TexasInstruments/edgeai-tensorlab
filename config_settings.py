from jacinto_ai_benchmark.defaults import *


############################################################
# common settings
verbose_mode = False
num_frames = 10000 #50000
max_frames_calib = 50 #100
max_calib_iterations = 50
modelzoo_path = '../jacinto-ai-modelzoo/models'
datasets_path = f'./dependencies/datasets'
target_device = 'pc'
parallel_devices = None #[0,1,2,3,0,1,2,3] #None
tidl_tensor_bits = 32 #8 #16 #32
run_import = True #False #True
run_inference = True #False #True
detection_thr = 0.05
save_output = True


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
imagenet_cls_train_cfg = dict(
    path=f'{datasets_path}/imagenet/val',
    split=f'{datasets_path}/imagenet/val.txt',
    shuffle=True,
    num_frames=quantization_params.get_num_frames_calib())

imagenet_cls_val_cfg = dict(
    path=f'{datasets_path}/imagenet/val',
    split=f'{datasets_path}/imagenet/val.txt',
    shuffle=True,
    num_frames=min(num_frames,50000))


coco_det_train_cfg = dict(
    path=f'{datasets_path}/coco',
    split='val2017',
    shuffle=True,
    num_frames=quantization_params.get_num_frames_calib())

coco_det_val_cfg = dict(
    path=f'{datasets_path}/coco',
    split='val2017',
    shuffle=False, #TODO: need to make COCODetection.evaluate() work with shuffle
    num_frames=min(num_frames,5000))


cityscapes_seg_train_cfg = dict(
    path=f'{datasets_path}/cityscapes',
    split='val',
    shuffle=True,
    num_frames=quantization_params.get_num_frames_calib())

cityscapes_seg_val_cfg = dict(
    path=f'{datasets_path}/cityscapes',
    split='val',
    shuffle=True,
    num_frames=min(num_frames,500))



