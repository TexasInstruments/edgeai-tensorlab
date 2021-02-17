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

import config_settings as settings

work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(__file__))[0], f'{settings.tidl_tensor_bits}bits')
print(f'work_dir = {work_dir}')


################################################################################################
# configs for each model pipeline
common_cfg = {
    'type': 'accuracy',
    'verbose': settings.verbose,
    'run_import': settings.run_import,
    'run_inference': settings.run_inference,
    'calibration_dataset': datasets.ImageNetCls(**settings.imagenet_cls_calib_cfg),
    'input_dataset': datasets.ImageNetCls(**settings.imagenet_cls_val_cfg),
    'postprocess': settings.get_postproc_classification()
}

common_session_cfg = dict(work_dir=work_dir, target_device=settings.target_device)

pipeline_configs = {
    # mobilenet_v2_2019-12-24_15-32-12 72.13% top-1 accuracy
    'eg1':utils.dict_update(common_cfg,
        preprocess=settings.get_preproc_onnx(),
        session=sessions.TVMDLRSession(**common_session_cfg, **settings.session_tvm_dlr_cfg,
            model_path=f'./dependencies/examples/models/mobilenet_v2_2019-12-24_15-32-12_opset9.onnx')
    ),
    # mlperf_mobilenet_v1_1.0_224 71.646% top-1 accuracy
    'eg2':utils.dict_update(common_cfg,
        preprocess=settings.get_preproc_tflite(),
        session=sessions.TFLiteRTSession(**common_session_cfg, **settings.session_tflite_rt_cfg,
            model_path=f'{settings.modelzoo_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v1_1.0_224.tflite'),
        metric=dict(label_offset_pred=-1)
    )
}


################################################################################################
# execute each model
if __name__ == '__main__':
    if settings.run_inference:
        pipelines.run(pipeline_configs, parallel_devices=settings.parallel_devices)
    #
    if settings.collect_results:
        results = pipelines.collect_results(work_dir)
        print(*results, sep='\n')
    #


