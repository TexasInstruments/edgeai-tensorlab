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

import default_settings as defaults
work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(__file__))[0])


################################################################################################
# configs for each model pipeline
pipeline_cfg = dict(
    type='accuracy',
    calibration_dataset=datasets.ImageNetCls(**defaults.imagenet_train_cfg),
    input_dataset=datasets.ImageNetCls(**defaults.imagenet_val_cfg),
    postprocess=defaults.get_postproc_classification()
)

pipeline_configs = [
    # mobilenet_v2_2019-12-24_15-32-12 72.13% top-1 accuracy
    utils.dict_update(pipeline_cfg,
        preprocess=defaults.get_preproc_tvm_dlr(),
        session=sessions.TVMDLRSession(**defaults.session_tvm_dlr_cfg, work_dir=work_dir,
            model_path=f'./dependencies/examples/models/mobilenet_v2_2019-12-24_15-32-12_opset9.onnx',
            input_shape={'input.1': (1, 3, 224, 224)})),
    # mlperf_mobilenet_v1_1.0_224 71.646% top-1 accuracy
    utils.dict_update(pipeline_cfg,
        preprocess=defaults.get_preproc_tflite_rt(),
        session=sessions.TFLiteRTSession(**defaults.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{defaults.modelzoo_path}/mlperf/edge/mlperf_mobilenet_v1_1.0_224.tflite',
            input_shape={'input': (1, 224, 224, 3)}),
        metric=dict(label_offset_pred=-1))
]


################################################################################################
# execute each model
if __name__ == '__main__':
    pipelines.run(pipeline_configs, devices=defaults.cuda_devices)
    results = pipelines.collect_results(work_dir)
    print(*results, sep='\n')


