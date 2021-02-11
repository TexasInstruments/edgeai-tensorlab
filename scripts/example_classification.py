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
work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(__file__))[0])


################################################################################################
# configs for each model pipeline
common_cfg = {
    'type':'accuracy',
    'target_device': config.target_device,
    'calibration_dataset':datasets.ImageNetCls(**config.imagenet_cls_calib_cfg),
    'input_dataset':datasets.ImageNetCls(**config.imagenet_cls_val_cfg),
    'postprocess':config.get_postproc_classification()
}

common_session_cfg = dict(work_dir=work_dir, target_device=config.target_device)

pipeline_configs = [
    # mobilenet_v2_2019-12-24_15-32-12 72.13% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_onnx(),
        'session':sessions.TVMDLRSession(**common_session_cfg, **config.session_tvm_dlr_cfg,
            model_path=f'./dependencies/examples/models/mobilenet_v2_2019-12-24_15-32-12_opset9.onnx',
            input_shape={'input.1': (1, 3, 224, 224)})
    }),
    # mlperf_mobilenet_v1_1.0_224 71.646% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_tflite(),
        'session':sessions.TFLiteRTSession(**common_session_cfg, **config.session_tflite_rt_cfg,
            model_path=f'{config.modelzoo_path}/vision/cls/imagenet1k/tf1-models/mobilenet_v1_1.0_224.tflite',
            input_shape={'input': (1, 224, 224, 3)}),
        'metric':dict(label_offset_pred=-1)
    })
]


################################################################################################
# execute each model
if __name__ == '__main__':
    if config.run_inference:
        pipelines.run(pipeline_configs, parallel_devices=config.parallel_devices)
    #
    results = pipelines.collect_results(work_dir)
    print(*results, sep='\n')


