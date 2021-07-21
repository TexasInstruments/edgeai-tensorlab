# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from jai_benchmark import constants, utils, datasets, preprocess, sessions, postprocess, metrics


def get_configs(settings, work_dir):
    # get the sessions types to use for each model type
    onnx_session_type = settings.get_session_type(constants.MODEL_TYPE_ONNX)
    tflite_session_type = settings.get_session_type(constants.MODEL_TYPE_TFLITE)
    mxnet_session_type = settings.get_session_type(constants.MODEL_TYPE_MXNET)

    # for this model, layer 43 is forced to ddr - this is a temporary fix
    runtime_options_onnx_ssd_np2 = settings.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=False,
                runtime_options={'ti_internal_reserved_1': 43})

    runtime_options_tflite_np2 = settings.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=False,
                runtime_options={'object_detection:score_threshold': settings.detection_thr})

    # configs for each model pipeline
    common_cfg = {
        'task_type': 'detection',
        'calibration_dataset': settings.dataset_cache['coco']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['coco']['input_dataset'],
    }

    common_session_cfg = dict(work_dir=work_dir, target_device=settings.target_device)

    postproc_detection_onnx = settings.get_postproc_detection_onnx()
    postproc_detection_tflite = settings.get_postproc_detection_tflite()
    postproc_detection_efficientdet_ti_lite_tflite = settings.get_postproc_detection_tflite(normalized_detections=False, ignore_detection_element=0,
                                                            formatter=postprocess.DetectionFormatting(dst_indices=(0,1,2,3,4,5), src_indices=(1,0,3,2,5,4)),
                                                            )
    postproc_detection_mxnet = settings.get_postproc_detection_mxnet()

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################onnx models#####################################
        # mlperf edge: detection - coco_ssd-resnet34_1200x1200 - expected_metric: 20.0% COCO AP[0.5-0.95]
        'vdet-12-012-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((1200,1200), (1200,1200), backend='cv2'),
            session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx_ssd_np2,
                model_path=f'{settings.models_path}/vision/detection/coco/mlperf/ssd_resnet34-ssd1200.onnx'),
            postprocess=postproc_detection_onnx,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=0)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':20.0})
        ),
        'vdet-12-100-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((512,512), (512,512), backend='cv2'),
            session=onnx_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), {'object_detection:meta_arch_type': 3, 'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd-lite_mobilenetv2_512x512_20201214_model.prototxt'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd-lite_mobilenetv2_512x512_20201214_model.onnx'),
            postprocess=settings.get_postproc_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':25.1})
        ),
        'vdet-12-101-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((512,512), (512,512), backend='cv2'),
            session=onnx_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), {'object_detection:meta_arch_type': 3, 'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd-lite_mobilenetv2_fpn_512x512_20201110_model.prototxt'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd-lite_mobilenetv2_fpn_512x512_20201110_model.onnx'),
            postprocess=settings.get_postproc_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':27.2})
        ),
        'vdet-12-102-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((320,320), (320,320), backend='cv2', reverse_channels=True),
            session=onnx_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), {'object_detection:meta_arch_type': 3, 'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd-lite_regnetx-200mf_fpn_bgr_320x320_20201010_model.prototxt'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd-lite_regnetx-200mf_fpn_bgr_320x320_20201010_model.onnx'),
            postprocess=settings.get_postproc_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':20.7})
        ),
        'vdet-12-103-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((512,512), (512,512), backend='cv2', reverse_channels=True),
            session=onnx_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), {'object_detection:meta_arch_type': 3, 'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd-lite_regnetx-800mf_fpn_bgr_512x512_20200919_model.prototxt'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd-lite_regnetx-800mf_fpn_bgr_512x512_20200919_model.onnx'),
            postprocess=settings.get_postproc_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':32.8})
        ),
        'vdet-12-104-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((768,768), (768,768), backend='cv2'),
            session=onnx_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), {'object_detection:meta_arch_type': 3, 'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd-lite_regnetx-1.6gf_bifpn168x4_bgr_768x768_20201026_model.prototxt'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd-lite_regnetx-1.6gf_bifpn168x4_bgr_768x768_20201026_model.onnx'),
            postprocess=settings.get_postproc_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':39.8})
        ),
        'vdet-12-105-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((416,416), (416,416), backend='cv2'),
            session=onnx_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(),
                                    {'object_detection:meta_arch_type': 4,
                                     'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolov3_d53_relu_416x416_20210117_004118_model.prototxt',
                                     'advanced_options:output_feature_16bit_names_list':'694, 698, 702'
                                    }),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolov3_d53_relu_416x416_20210117_004118_model.onnx'),
            postprocess=settings.get_postproc_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':30.7})
        ),
        'vdet-12-106-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((512,512), (512,512), backend='cv2'),
            session=onnx_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(),
                                    {'object_detection:meta_arch_type': 4,
                                     'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolov3-lite_regnetx-1.6gf_bgr_512x512_20210202_model.prototxt',
                                     'advanced_options:output_feature_16bit_names_list':'823, 830, 837'
                                    }),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolov3-lite_regnetx-1.6gf_bgr_512x512_20210202_model.onnx'),
            postprocess=settings.get_postproc_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':30.7})
        ),
        'vdet-12-107-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((512,512), (512,512), backend='cv2'),
            session=onnx_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), {'object_detection:meta_arch_type': 5, 'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/retinanet-lite_regnetx-800mf_fpn_bgr_512x512_20200908_model.prototxt'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/retinanet-lite_regnetx-800mf_fpn_bgr_512x512_20200908_model.onnx'),
            postprocess=settings.get_postproc_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':33.0})
        ),
        #################################################################
        #       MXNET MODELS
        #################################################################
        # mxnet : gluoncv model : detection - yolo3_mobilenet1.0_coco - accuracy: 28.6% ap[0.5:0.95], 48.9% ap50
        'vdet-12-060-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((416,416), (416,416), backend='cv2'),
            session=mxnet_session_type(**common_session_cfg, runtime_options=settings.runtime_options_mxnet_np2(),
                model_path=[f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/yolo3_mobilenet1.0_coco-symbol.json',
                            f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/yolo3_mobilenet1.0_coco-0000.params'],
                model_type='mxnet', input_shape={'data':(1,3,416,416)}),
            postprocess=postproc_detection_mxnet,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':28.6})
        ),
        # mxnet : gluoncv model : detection - ssd_512_resnet50_v1_coco - accuracy: 30.6% ap[0.5:0.95], 50.0% ap50
        'vdet-12-061-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((512,512), (512,512), backend='cv2'),
            session=mxnet_session_type(**common_session_cfg, runtime_options=settings.runtime_options_mxnet_p2(),
                model_path=[f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_resnet50_v1_coco-symbol.json',
                            f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_resnet50_v1_coco-0000.params'],
                model_type='mxnet', input_shape={'data':(1,3,512,512)}),
            postprocess=postproc_detection_mxnet,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':30.6})
        ),
        # mxnet : gluoncv model : detection - ssd_512_mobilenet1.0_coco - accuracy: 21.7% ap[0.5:0.95], 39.2% ap50
        'vdet-12-062-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx((512,512), (512,512), backend='cv2'),
            session=mxnet_session_type(**common_session_cfg, runtime_options=settings.runtime_options_mxnet_np2(),
                model_path=[f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_mobilenet1.0_coco-symbol.json',
                            f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_mobilenet1.0_coco-0000.params'],
                model_type='mxnet', input_shape={'data':(1,3,512,512)}),
            postprocess=postproc_detection_mxnet,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':21.7})
        ),
        #################################################################
        #       TFLITE MODELS
        #################tflite models###################################
        # mlperf edge: detection - ssd_mobilenet_v1_coco_2018_01_28 expected_metric: 23.0% ap[0.5:0.95] accuracy
        'vdet-12-010-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((300,300), (300,300), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_np2,
                model_path=f'{settings.models_path}/vision/detection/coco/mlperf/ssd_mobilenet_v1_coco_20180128.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':23.0})
        ),
        # mlperf mobile: detection - ssd_mobilenet_v2_coco_300x300 - expected_metric: 22.0% COCO AP[0.5-0.95]
        'vdet-12-011-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((300,300), (300,300), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_np2,
                model_path=f'{settings.models_path}/vision/detection/coco/mlperf/ssd_mobilenet_v2_300_float.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':22.0})
        ),
        #################################################################
        # tensorflow1.0 models: detection - ssdlite_mobiledet_dsp_320x320_coco_2020_05_19 expected_metric: 28.9% ap[0.5:0.95] accuracy
        'vdet-12-400-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((320,320), (320,320), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_np2,
                model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':28.9})
        ),
        # tensorflow1.0 models: detection - ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19 expected_metric: 25.9% ap[0.5:0.95] accuracy
        'vdet-12-401-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((320,320), (320,320), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_np2,
                model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssdlite_mobiledet_edgetpu_320x320_coco_20200519.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':25.9})
        ),
        # tensorflow1.0 models: detection - ssdlite_mobilenet_v2_coco_2018_05_09 expected_metric: 22.0% ap[0.5:0.95] accuracy
        'vdet-12-402-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((300,300), (300,300), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_np2,
                model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssdlite_mobilenet_v2_coco_20180509.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':22.0})
        ),
        'vdet-12-403-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((640,640), (640,640), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_np2,
                model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_20180703.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':32.0})
        ),
        'vdet-12-404-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((320,320), (320,320), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_np2,
                model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_20200518.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':26.6})
        ),
        'vdet-12-410-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((640,640), (640,640), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_np2,
                model_path=f'{settings.models_path}/vision/detection/coco/tf2-models/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':29.1})
        ),
        'vdet-12-411-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((320,320), (320,320), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_np2,
                model_path=f'{settings.models_path}/vision/detection/coco/tf2-models/ssd_mobilenet_v2_320x320_coco17_tpu-8.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':20.2})
        ),
        'vdet-12-412-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((320,320), (320,320), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_np2,
                model_path=f'{settings.models_path}/vision/detection/coco/tf2-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':22.2})
        ),
        'vdet-12-413-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((640,640), (640,640), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_np2,
                model_path=f'{settings.models_path}/vision/detection/coco/tf2-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':28.2})
        ),
        'vdet-12-414-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((640,640), (640,640), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_np2,
                model_path=f'{settings.models_path}/vision/detection/coco/tf2-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':34.3})
        ),
        'vdet-12-420-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite((512,512), (512,512), backend='cv2',mean=(123.675, 116.28, 103.53), scale=(0.01712475, 0.017507, 0.01742919)),
            session=tflite_session_type(**common_session_cfg,
                runtime_options=utils.dict_update(runtime_options_tflite_np2, {'object_detection:meta_arch_type': 5, 'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/google-automl/efficientdet-lite0_bifpn_maxpool2x2_relu_ti-lite.prototxt'}),
                model_path=f'{settings.models_path}/vision/detection/coco/google-automl/efficientdet-lite0_bifpn_maxpool2x2_relu_ti-lite.tflite'),
            postprocess=postproc_detection_efficientdet_ti_lite_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90(label_offset=0)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':33.61})
        ),
    }
    return pipeline_configs

