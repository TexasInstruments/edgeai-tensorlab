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
    'type': settings.type,
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
            model_path=f'./dependencies/examples/models/mobilenet_v2_20191224-153212_opset9.onnx')
    ),
    # tensorflow/models: classification mobilenetv1_224x224 expected_metric: 71.0% top-1 accuracy
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
    pipeline_runner = pipelines.PipelineRunner(settings, pipeline_configs)
    
    if settings.run_import or settings.run_inference:
        pipeline_runner.run()
    #

    if settings.collect_results:
        results = pipelines.collect_results(work_dir)
        print(*results, sep='\n')
    #


