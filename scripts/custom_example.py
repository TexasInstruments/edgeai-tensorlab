import os
import argparse
from jacinto_ai_benchmark import *

# the cwd must be the root of the respository
if os.path.split(os.getcwd())[-1] == 'scripts':
    os.chdir('../')
#

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', type=str)
    cmds = parser.parse_args()
    settings = config_settings.ConfigSettings(cmds.settings_file)

    expt_name = os.path.splitext(os.path.basename(__file__))[0]
    work_dir = os.path.join('./work_dirs', expt_name, f'{settings.tidl_tensor_bits}bits')
    print(f'work_dir = {work_dir}')

    ################################################################################################
    # configs for each model pipeline
    common_cfg = {
        'pipeline_type': settings.pipeline_type,
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
        'example1':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx(),
            session=sessions.TVMDLRSession(**common_session_cfg, **settings.session_tvm_dlr_cfg,
                model_path=f'./dependencies/examples/models/mobilenet_v2_20191224-153212_opset9.onnx')
        ),
        # tensorflow/models: classification mobilenetv1_224x224 expected_metric: 71.0% top-1 accuracy
        'example2':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite(),
            session=sessions.TFLiteRTSession(**common_session_cfg, **settings.session_tflite_rt_cfg,
                model_path=f'{settings.modelzoo_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v1_1.0_224.tflite'),
            metric=dict(label_offset_pred=-1)
        )
    }

    ################################################################################################
    # create runner and run the pipeline
    pipeline_runner = pipelines.PipelineRunner(settings, pipeline_configs)
    
    # now actually run the configs
    if settings.run_import or settings.run_inference:
        pipeline_runner.run()
    #

    # collect the logs and display it
    if settings.collect_results:
        results = pipelines.collect_results(settings, work_dir, print_results=True)
    #


