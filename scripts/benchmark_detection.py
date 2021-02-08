import os
from jacinto_ai_benchmark import *

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
    'run_import':config.run_import,
    'run_inference':config.run_inference,
    'calibration_dataset':datasets.COCODetection(**config.coco_det_train_cfg),
    'input_dataset':datasets.COCODetection(**config.coco_det_val_cfg),
    'postprocess':config.get_postproc_detection()
}

pipeline_configs = [
    #################################################################
    #       ONNX MODELS
    #################jai-devkit models##############################
    # jai-devkit: classification mobilenetv1_224x224 expected_metric: 71.82% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_vgg(),
        'session':sessions.TFLiteRTSession(**config.session_tvm_dlr_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/detection/coco/mlperf/ssd_mobilenet_v1_coco_2018_01_28.tflite')
    }),
]


################################################################################################
# execute each model
if __name__ == '__main__':
    if config.run_import or config.run_inference:
        pipelines.run(pipeline_configs, devices=config.cuda_devices)
    #
    results = pipelines.collect_results(work_dir)
    print(*results, sep='\n')


