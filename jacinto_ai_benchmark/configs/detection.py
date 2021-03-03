from .. import utils, datasets, preprocess, sessions, postprocess, metrics


def get_configs(settings, work_dir):
    # get the sessions types to use for each model type
    session_type_dict = sessions.convert_session_names_to_types(settings.session_type_dict)
    onnx_session_type = session_type_dict['onnx']
    tflite_session_type = session_type_dict['tflite']
    mxnet_session_type = session_type_dict['mxnet']

    # configs for each model pipeline
    common_cfg = {
        'pipeline_type': settings.pipeline_type,
        'verbose': settings.verbose,
        'target_device': settings.target_device,
        'run_import': settings.run_import,
        'run_inference': settings.run_inference,
        'calibration_dataset': settings.dataset_cache['coco']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['coco']['input_dataset'],
    }

    common_session_cfg = dict(work_dir=work_dir, target_device=settings.target_device)

    postproc_detection_onnx = settings.get_postproc_detection_onnx()
    postproc_detection_tflite = settings.get_postproc_detection_tflite()
    postproc_detection_mxnet = settings.get_postproc_detection_mxnet()

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################onnx models#####################################
        # # mlperf edge: detection - coco_ssd-resnet34_1200x1200 - expected_metric: 20.0% COCO AP[0.5-0.95]
        # 'vdet-12-012-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_onnx((1200,1200), (1200,1200), backend='cv2'),
        #     session=onnx_session_type(**common_session_cfg, **settings.session_tvm_dlr_cfg,
        #         model_path=f'{settings.modelzoo_path}/vision/detection/coco/mlperf/ssd_resnet34-ssd1200.onnx'),
        #     postprocess=postproc_detection_onnx,
        #     metric=dict(label_offset_pred=det_helper.coco_label_offset_80to90())
        # ),
        #################################################################
        # # yolov3: detection - yolov3 416x416 - expected_metric: 31.0% COCO AP[0.5-0.95]
        # 'vdet-12-020-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_onnx((416,416), (416,416), backend='cv2',
        #         mean=(0.0, 0.0, 0.0), scale=(1/255.0, 1/255.0, 1/255.0)),
        #     session=onnx_session_type(**common_session_cfg, **settings.session_tvm_dlr_cfg,
        #         model_path=f'{settings.modelzoo_path}/vision/detection/coco/onnx-models/yolov3-10.onnx',
        #         input_shape=dict(input_1=(1,3,416,416), image_shape=(1,2))),
        #     postprocess=postproc_detection_onnx,
        #     metric=dict(label_offset_pred=det_helper.coco_label_offset_80to90())
        # ),
        #################################################################
        #       TFLITE MODELS
        #################tflite models###################################
        # mlperf edge: detection - ssd_mobilenet_v1_coco_2018_01_28 expected_metric: 23.0% ap[0.5:0.95] accuracy
        'vdet-12-010-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((300,300), (300,300), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, **settings.session_tflite_rt_cfg,
                model_path=f'{settings.modelzoo_path}/vision/detection/coco/mlperf/ssd_mobilenet_v1_coco_2018_01_28.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=coco_label_offset_90to90())
        ),
        # mlperf mobile: detection - ssd_mobilenet_v2_coco_300x300 - expected_metric: 22.0% COCO AP[0.5-0.95]
        'vdet-12-011-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((300,300), (300,300), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, **settings.session_tflite_rt_cfg,
                model_path=f'{settings.modelzoo_path}/vision/detection/coco/mlperf/ssd_mobilenet_v2_300_float.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=coco_label_offset_90to90())
        ),
        #################################################################
        # tensorflow1.0 models: detection - ssdlite_mobiledet_dsp_320x320_coco_2020_05_19 expected_metric: 28.9% ap[0.5:0.95] accuracy
        'vdet-12-400-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((320,320), (320,320), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, **settings.session_tflite_rt_cfg,
                model_path=f'{settings.modelzoo_path}/vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_2020_05_19.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=coco_label_offset_90to90())
        ),
        # tensorflow1.0 models: detection - ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19 expected_metric: 25.9% ap[0.5:0.95] accuracy
        'vdet-12-401-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((320,320), (320,320), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, **settings.session_tflite_rt_cfg,
                model_path=f'{settings.modelzoo_path}/vision/detection/coco/tf1-models/ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=coco_label_offset_90to90())
        ),
        # tensorflow1.0 models: detection - ssdlite_mobilenet_v2_coco_2018_05_09 expected_metric: 22.0% ap[0.5:0.95] accuracy
        'vdet-12-402-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((300,300), (300,300), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, **settings.session_tflite_rt_cfg,
                model_path=f'{settings.modelzoo_path}/vision/detection/coco/tf1-models/ssdlite_mobilenet_v2_coco_2018_05_09.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=coco_label_offset_90to90())
        ),
        # # tensorflow1.0 models: detection - ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18 expected_metric: 26.6% ap[0.5:0.95] accuracy
        # 'vdet-12-403-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_tflite((320,320), (320,320), backend='cv2'),
        #     session=tflite_session_type(**common_session_cfg, **settings.session_tflite_rt_cfg,
        #         model_path=f'{settings.modelzoo_path}/vision/detection/coco/tf1-models/ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_2020_05_18.tflite'),
        #     postprocess=postproc_detection_tflite,
        #     metric=dict(label_offset_pred=coco_label_offset_90to90())
        # ),
        # # tensorflow1.0 models: detection - ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03 expected_metric: 32.0% ap[0.5:0.95] accuracy
        # 'vdet-12-404-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_tflite((640,640), (640,640), backend='cv2'),
        #     session=tflite_session_type(**common_session_cfg, **settings.session_tflite_rt_cfg,
        #         model_path=f'{settings.modelzoo_path}/vision/detection/coco/tf1-models/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tflite'),
        #     postprocess=postproc_detection_tflite,
        #     metric=dict(label_offset_pred=coco_label_offset_90to90())
        # ),

        #################################################################
        # tensorflow2.0 models: detection - ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8 expected_metric: 28.2% ap[0.5:0.95] accuracy
        # 'vdet-12-450-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_tflite((640,640), (640,640), backend='cv2'),
        #     session=tflite_session_type(**common_session_cfg, **settings.session_tflite_rt_cfg,
        #         model_path=f'{settings.modelzoo_path}/vision/detection/coco/tf2-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tflite'),
        #     postprocess=postproc_detection_tflite,
        #     metric=dict(label_offset_pred=coco_label_offset_90to90())
        # ),
        # # tensorflow2.0 models: detection - ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 expected_metric: 34.3% ap[0.5:0.95] accuracy
        # 'vdet-12-451-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_tflite((640,640), (640,640), backend='cv2'),
        #     session=tflite_session_type(**common_session_cfg, **settings.session_tflite_rt_cfg,
        #         model_path=f'{settings.modelzoo_path}/vision/detection/coco/tf2-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tflite'),
        #     postprocess=postproc_detection_tflite,
        #     metric=dict(label_offset_pred=coco_label_offset_90to90())
        # ),
        # # tensorflow2.0 models: detection - ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8 expected_metric: 38.3% ap[0.5:0.95] accuracy
        # 'vdet-12-452-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_tflite((1024,1024), (1024,1024), backend='cv2'),
        #     session=tflite_session_type(**common_session_cfg, **settings.session_tflite_rt_cfg,
        #         model_path=f'{settings.modelzoo_path}/vision/detection/coco/tf2-models/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tflite'),
        #     postprocess=postproc_detection_tflite,
        #     metric=dict(label_offset_pred=coco_label_offset_90to90())
        # ),
        #################################################################
        # # google automl: detection - efficientdet-lite0_bifpn_maxpool2x2_relu expected_metric: 33.5% ap[0.5:0.95] accuracy
        # 'vdet-12-040-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_tflite((512,512), (512,512), backend='cv2'),
        #     session=tflite_session_type(**common_session_cfg, **settings.session_tflite_rt_cfg,
        #         model_path=f'{settings.modelzoo_path}/vision/detection/coco/google-automl/efficientdet-lite0_bifpn_maxpool2x2_relu.tflite'),
        #     postprocess=postproc_detection_tflite,
        #     metric=dict(label_offset_pred=coco_label_offset_90to90())
        # ),
        #################################################################
        # mxnet : gluoncv model : detection - yolo3_mobilenet1.0_coco
        'vdet-12-060-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((416,416), (416,416), backend='cv2'),
            session=mxnet_session_type(**common_session_cfg, **settings.session_tvm_dlr_cfg,
                model_path=[f'{settings.modelzoo_path}/vision/detection/coco/gluoncv-mxnet/yolo3_mobilenet1.0_coco-symbol.json',
                            f'{settings.modelzoo_path}/vision/detection/coco/gluoncv-mxnet/yolo3_mobilenet1.0_coco-0000.params'],
                model_type='mxnet', input_shape={'data':(1,3,416,416)}),
            postprocess=postproc_detection_mxnet,
            metric=dict(label_offset_pred=coco_label_offset_80to90())
        ),
        # mxnet : gluoncv model : detection - ssd_512_mobilenet1.0_coco
        'vdet-12-061-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((512,512), (512,512), backend='cv2'),
            session=mxnet_session_type(**common_session_cfg, **settings.session_tvm_dlr_cfg,
                model_path=[f'{settings.modelzoo_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_mobilenet1.0_coco-symbol.json',
                            f'{settings.modelzoo_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_mobilenet1.0_coco-0000.params'],
                model_type='mxnet', input_shape={'data':(1,3,512,512)}),
            postprocess=postproc_detection_mxnet,
            metric=dict(label_offset_pred=coco_label_offset_80to90())
        ),
    }
    return pipeline_configs



################################################################################################
# convert from 80 class index (typical output of a mmdetection detector) to 90 or 91 class
# (original labels of coco starts from 1, and 0 is background)
# the evalation/metric script will convert from 80 class to coco's 90 class.
def coco_label_offset_80to90(label_offset=1):
    coco_label_table = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                         21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                         41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                         61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                         81, 82, 84, 85, 86, 87, 88, 89, 90]

    if label_offset == 1:
        # 0 => 1, 1 => 2, .. 79 -> 90, 80 => 91
        coco_label_offset = {k:v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({80:91})
    elif label_offset == 0:
        # 0 => 0, 1 => 1, .. 80 => 90
        coco_label_offset = {(k+1):v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({0:0})
    else:
        assert False, f'unsupported value for label_offset {label_offset}'
    #
    return coco_label_offset


# convert from 90 class index (typical output of a tensorflow detector) to 90 or 91 class
# (original labels of coco starts from 1, and 0 is background)
def coco_label_offset_90to90(label_offset=1):
    coco_label_table = range(1,91)
    if label_offset == 1:
        # 0 => 1, 1 => 2, .. 90 => 91
        coco_label_offset = {k:v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({-1:0,90:91})
    elif label_offset == 0:
        # 0 => 0, 1 => 1, .. 90 => 90
        coco_label_offset = {(k+1):v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({-1:-1,0:0})
    else:
        assert False, f'unsupported value for label_offset {label_offset}'
    #
    return coco_label_offset