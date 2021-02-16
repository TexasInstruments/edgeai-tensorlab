import os
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
    'verbose': config.verbose,
    'target_device': config.target_device,
    'run_import':config.run_import,
    'run_inference':config.run_inference,
    'calibration_dataset':datasets.COCODetection(**config.coco_det_calib_cfg),
    'input_dataset':datasets.COCODetection(**config.coco_det_val_cfg),
}

common_session_cfg = dict(work_dir=work_dir, target_device=config.target_device)

postproc_detection_onnx = config.get_postproc_detection_onnx(score_thr=config.detection_thr, save_output=config.save_output)
postproc_detection_tflite = config.get_postproc_detection_tflite(score_thr=config.detection_thr, save_output=config.save_output)

pipeline_configs = [
    #################################################################
    #       ONNX MODELS
    #################onnx models#####################################
    # # mlperf edge: detection - coco_ssd-resnet34_1200x1200 - expected_metric: 20.0% COCO AP[0.5-0.95]
    # utils.dict_update(common_cfg, {
    #     'preprocess':config.get_preproc_tflite((1200,1200), (1200,1200), backend='cv2'),
    #     'session':sessions.TVMDLRSession(**common_session_cfg, **config.session_tvm_dlr_cfg,
    #         model_path=f'{config.modelzoo_path}/vision/detection/coco/mlperf/ssd_resnet34-ssd1200.onnx'),
    #     'postprocess': postproc_detection_tflite,
    #     'metric':dict(label_offset_pred=det_helper.coco_label_offset_80to90())
    # }),
    #################################################################
    # # yolov3: detection - yolov3 416x416 - expected_metric: 31.0% COCO AP[0.5-0.95]
    # utils.dict_update(common_cfg, {
    #     'preprocess':config.get_preproc_onnx((416,416), (416,416), backend='cv2',
    #         mean=(0.0, 0.0, 0.0), scale=(1/255.0, 1/255.0, 1/255.0)),
    #     'session':sessions.TVMDLRSession(**common_session_cfg, **config.session_tvm_dlr_cfg,
    #         model_path=f'{config.modelzoo_path}/vision/detection/coco/onnx-models/yolov3-10.onnx',
    #         input_shape=dict(input_1=(1,3,416,416), image_shape=(1,2))),
    #     'postprocess': postproc_detection_onnx,
    #     'metric':dict(label_offset_pred=det_helper.coco_label_offset_80to90())
    # }),
    #################################################################
    #       TFLITE MODELS
    #################tflite models###################################
    # mlperf edge: detection - ssd_mobilenet_v1_coco_2018_01_28 expected_metric: 23.0% ap[0.5:0.95] accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_tflite((300,300), (300,300), backend='cv2'),
        'session':sessions.TFLiteRTSession(**common_session_cfg, **config.session_tflite_rt_cfg,
            model_path=f'{config.modelzoo_path}/vision/detection/coco/mlperf/ssd_mobilenet_v1_coco_2018_01_28.tflite'),
        'postprocess': postproc_detection_tflite,
        'metric':dict(label_offset_pred=det_helper.coco_label_offset_90to90())
    }),
    # mlperf mobile: detection - ssd_mobilenet_v2_coco_300x300 - expected_metric: 22.0% COCO AP[0.5-0.95]
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_tflite((300,300), (300,300), backend='cv2'),
        'session':sessions.TFLiteRTSession(**common_session_cfg, **config.session_tflite_rt_cfg,
            model_path=f'{config.modelzoo_path}/vision/detection/coco/mlperf/ssd_mobilenet_v2_300_float.tflite'),
        'postprocess': postproc_detection_tflite,
        'metric':dict(label_offset_pred=det_helper.coco_label_offset_90to90())
    }),
    #################################################################
    # tensorflow1.0 models: detection - ssdlite_mobiledet_dsp_320x320_coco_2020_05_19 expected_metric: 28.9% ap[0.5:0.95] accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_tflite((320,320), (320,320), backend='cv2'),
        'session':sessions.TFLiteRTSession(**common_session_cfg, **config.session_tflite_rt_cfg,
            model_path=f'{config.modelzoo_path}/vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_2020_05_19.tflite'),
        'postprocess': postproc_detection_tflite,
        'metric':dict(label_offset_pred=det_helper.coco_label_offset_90to90())
    }),
    # tensorflow1.0 models: detection - ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19 expected_metric: 25.9% ap[0.5:0.95] accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_tflite((320,320), (320,320), backend='cv2'),
        'session':sessions.TFLiteRTSession(**common_session_cfg, **config.session_tflite_rt_cfg,
            model_path=f'{config.modelzoo_path}/vision/detection/coco/tf1-models/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tflite'),
        'postprocess': postproc_detection_tflite,
        'metric':dict(label_offset_pred=det_helper.coco_label_offset_90to90())
    }),
    # tensorflow1.0 models: detection - ssdlite_mobilenet_v2_coco_2018_05_09 expected_metric: 22.0% ap[0.5:0.95] accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_tflite((300,300), (300,300), backend='cv2'),
        'session':sessions.TFLiteRTSession(**common_session_cfg, **config.session_tflite_rt_cfg,
            model_path=f'{config.modelzoo_path}/vision/detection/coco/tf1-models/ssdlite_mobilenet_v2_coco_2018_05_09.tflite'),
        'postprocess': postproc_detection_tflite,
        'metric':dict(label_offset_pred=det_helper.coco_label_offset_90to90())
    }),
    # tensorflow1.0 models: detection - ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18 expected_metric: 26.6% ap[0.5:0.95] accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_tflite((320,320), (320,320), backend='cv2'),
        'session':sessions.TFLiteRTSession(**common_session_cfg, **config.session_tflite_rt_cfg,
            model_path=f'{config.modelzoo_path}/vision/detection/coco/tf1-models/ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18.tflite'),
        'postprocess': postproc_detection_tflite,
        'metric':dict(label_offset_pred=det_helper.coco_label_offset_90to90())
    }),
    # tensorflow1.0 models: detection - ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03 expected_metric: 32.0% ap[0.5:0.95] accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_tflite((640,640), (640,640), backend='cv2'),
        'session':sessions.TFLiteRTSession(**common_session_cfg, **config.session_tflite_rt_cfg,
            model_path=f'{config.modelzoo_path}/vision/detection/coco/tf1-models/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tflite'),
        'postprocess': postproc_detection_tflite,
        'metric':dict(label_offset_pred=det_helper.coco_label_offset_90to90())
    }),

    #################################################################
    # tensorflow2.0 models: detection - ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8 expected_metric: 28.2% ap[0.5:0.95] accuracy
    # utils.dict_update(common_cfg, {
    #     'preprocess':config.get_preproc_tflite((640,640), (640,640), backend='cv2'),
    #     'session':sessions.TFLiteRTSession(**common_session_cfg, **config.session_tflite_rt_cfg,
    #         model_path=f'{config.modelzoo_path}/vision/detection/coco/tf2-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tflite'),
    #     'postprocess': postproc_detection_tflite,
    #     'metric':dict(label_offset_pred=det_helper.coco_label_offset_90to90())
    # }),
    # # tensorflow2.0 models: detection - ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 expected_metric: 34.3% ap[0.5:0.95] accuracy
    # utils.dict_update(common_cfg, {
    #     'preprocess':config.get_preproc_tflite((640,640), (640,640), backend='cv2'),
    #     'session':sessions.TFLiteRTSession(**common_session_cfg, **config.session_tflite_rt_cfg,
    #         model_path=f'{config.modelzoo_path}/vision/detection/coco/tf2-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tflite'),
    #     'postprocess': postproc_detection_tflite,
    #     'metric':dict(label_offset_pred=det_helper.coco_label_offset_90to90())
    # }),
    # # tensorflow2.0 models: detection - ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8 expected_metric: 38.3% ap[0.5:0.95] accuracy
    # utils.dict_update(common_cfg, {
    #     'preprocess':config.get_preproc_tflite((1024,1024), (1024,1024), backend='cv2'),
    #     'session':sessions.TFLiteRTSession(**common_session_cfg, **config.session_tflite_rt_cfg,
    #         model_path=f'{config.modelzoo_path}/vision/detection/coco/tf2-models/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tflite'),
    #     'postprocess': postproc_detection_tflite,
    #     'metric':dict(label_offset_pred=det_helper.coco_label_offset_90to90())
    # }),
    #################################################################
    # # google automl: detection - efficientdet-lite0_bifpn_maxpool2x2_relu expected_metric: 33.5% ap[0.5:0.95] accuracy
    # utils.dict_update(common_cfg, {
    #     'preprocess':config.get_preproc_tflite((512,512), (512,512), backend='cv2'),
    #     'session':sessions.TFLiteRTSession(**common_session_cfg, **config.session_tflite_rt_cfg,
    #         model_path=f'{config.modelzoo_path}/vision/detection/coco/google-automl/efficientdet-lite0_bifpn_maxpool2x2_relu.tflite'),
    #     'postprocess': postproc_detection_tflite,
    #     'metric':dict(label_offset_pred=det_helper.coco_label_offset_90to90())
    # }),
]


################################################################################################
# execute each model
if __name__ == '__main__':
    if config.run_import or config.run_inference:
        pipelines.run(config, pipeline_configs, parallel_devices=config.parallel_devices)
    #
    if config.collect_results:
        results = pipelines.collect_results(config, work_dir)
        print(*results, sep='\n')
    #


