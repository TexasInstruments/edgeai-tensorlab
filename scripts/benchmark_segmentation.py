import os
import cv2
from jacinto_ai_benchmark import *

import benchmark_detection_helper as det_helper

# the cwd must be the root of the respository
if os.path.split(os.getcwd())[-1] == 'scripts':
    os.chdir('../')
#
# make sure current directory is visible for python import
if not os.environ['PYTHONPATH'].startswith(':'):
    os.environ['PYTHONPATH'] = ':' + os.environ['PYTHONPATH']
#

import config_settings as config
work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(__file__))[0], f'{config.tidl_tensor_bits}bits')
print(f'work_dir = {work_dir}')


################################################################################################
# configs for each model pipeline
common_cfg = {
    'type':'accuracy',
    'verbose_mode': config.verbose_mode,
    'target_device': config.target_device,
    'run_import':config.run_import,
    'run_inference':config.run_inference,
    'calibration_dataset':datasets.CityscapesSegmentation(**config.cityscapes_seg_train_cfg),
    'input_dataset':datasets.CityscapesSegmentation(**config.cityscapes_seg_val_cfg),
}

common_session_cfg = dict(work_dir=work_dir, target_device=config.target_device)

postproc_segmentation_onnx = config.get_postproc_segmentation_onnx(save_output=config.save_output)
postproc_segmenation_tflite = config.get_postproc_segmentation_tflite(save_output=config.save_output)

pipeline_configs = [
    #################################################################
    #       TFLITE MODELS
    #################mlperf models##############################
    # mlperf edge: detection - ssd_mobilenet_v1_coco_2018_01_28 expected_metric: 23.0% ap[0.5:0.95] accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_jai((384,768), (384,768), backend='cv2', interpolation=cv2.INTER_AREA),
        'session':sessions.TVMDLRSession(**common_session_cfg, **config.session_tvm_dlr_cfg,
            model_path=f'{config.modelzoo_path}/vision/seg/cityscapes/jai-pytorch/deeplabv3lite_mobilenetv2_tv_768x384_20190626-085932_opset9.onnx'),
        'postprocess': postproc_segmentation_onnx
    }),
]


################################################################################################
# execute each model
if __name__ == '__main__':
    if config.run_import or config.run_inference:
        pipelines.run(pipeline_configs, parallel_devices=config.parallel_devices)
    #
    results = pipelines.collect_results(work_dir)
    print(*results, sep='\n')


