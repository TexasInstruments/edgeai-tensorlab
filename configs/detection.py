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
from edgeai_benchmark import constants, utils, datasets, preprocess, sessions, postprocess, metrics


def get_configs(settings, work_dir):
    # get the sessions types to use for each model type
    onnx_session_type = settings.get_session_type(constants.MODEL_TYPE_ONNX)
    tflite_session_type = settings.get_session_type(constants.MODEL_TYPE_TFLITE)
    mxnet_session_type = settings.get_session_type(constants.MODEL_TYPE_MXNET)

    preproc_transforms = preprocess.PreProcessTransforms(settings)
    postproc_transforms = postprocess.PostProcessTransforms(settings)

    # configs for each model pipeline
    common_cfg = {
        'task_type': 'detection',
        'dataset_category': datasets.DATASET_CATEGORY_COCO,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_COCO]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_COCO]['input_dataset'],
    }

    postproc_detection_onnx = postproc_transforms.get_transform_detection_onnx()
    postproc_detection_tflite = postproc_transforms.get_transform_detection_tflite()
    postproc_detection_efficientdet_ti_lite_tflite = postproc_transforms.get_transform_detection_tflite(normalized_detections=False, ignore_index=0,
                                                            formatter=postprocess.DetectionFormatting(dst_indices=(0,1,2,3,4,5), src_indices=(1,0,3,2,5,4)),
                                                            )
    postproc_detection_mxnet = postproc_transforms.get_transform_detection_mxnet()

    # reduce these iterations for slow models
    calibration_frames_fast = min(10, settings.calibration_frames)
    calibration_iterations_fast = min(5, settings.calibration_iterations)

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################onnx models#####################################
        # mlperf edge: detection - coco_ssd-resnet34_1200x1200 - expected_metric: 20.0% COCO AP[0.5-0.95]
        # for this model, layer 43 is forced to ddr - this is a temporary fix
        'od-8000':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((1200,1200), (1200,1200), backend='cv2'),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={'ti_internal_reserved_1': 43, 'object_detection:meta_arch_type': 3,
                   'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/mlperf/ssd_resnet34-ssd1200.prototxt',
                   'advanced_options:calibration_frames': calibration_frames_fast,
                   'advanced_options:calibration_iterations': calibration_iterations_fast,
                   }, fast_calibration=True),
                model_path=f'{settings.models_path}/vision/detection/coco/mlperf/ssd_resnet34-ssd1200.onnx'),
            postprocess=postproc_transforms.get_transform_detection_onnx(reshape_list=[(-1,4), (-1,1), (-1,1)], squeeze_axis=None),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=0)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':20.0}, model_shortlist=None, compact_name='ssd1200-resNet34-mlperf-coco-1200x1200', shortlisted=False)
        ),
        'od-8020':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((512,512), (512,512), backend='cv2'),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 3,
                     'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_lite_512x512_20201214_model.prototxt'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_lite_512x512_20201214_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':25.1}, model_shortlist=10, compact_name='ssd-lite-mobv2-mmdet-coco-512x512', shortlisted=True, recommended=True)
        ),
        'od-8030':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((512,512), (512,512), backend='cv2'),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 3,
                     'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_fpn_lite_512x512_20201110_model.prototxt'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_fpn_lite_512x512_20201110_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':27.2}, model_shortlist=None, compact_name='ssd-lite-mobv2-fpn-mmdet-coco-512x512', shortlisted=False)
        ),
        'od-8040':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((320,320), (320,320), backend='cv2', reverse_channels=True),
            session=onnx_session_type(**sessions.get_onnx_bgr_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 3,
                     'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd_regnetx-200mf_fpn_bgr_lite_320x320_20201010_model.prototxt'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd_regnetx-200mf_fpn_bgr_lite_320x320_20201010_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':20.7}, model_shortlist=30, compact_name='ssd-lite-regNetX-200mf-fpn-bgr-mmdet-coco-320x320', shortlisted=True)
        ),
        'od-8050':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((512,512), (512,512), backend='cv2', reverse_channels=True),
            session=onnx_session_type(**sessions.get_onnx_bgr_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 3,
                     'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd_regnetx-800mf_fpn_bgr_lite_512x512_20200919_model.prototxt'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd_regnetx-800mf_fpn_bgr_lite_512x512_20200919_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':32.8}, model_shortlist=None, compact_name='ssd-lite-regNetX-800mf-fpn-bgr-mmdet-coco-512x512', shortlisted=False)
        ),
        'od-8060':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((768,768), (768,768), backend='cv2'),
            session=onnx_session_type(**sessions.get_onnx_bgr_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 3,
                     'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd_regnetx-1.6gf_fpn_bgr_lite_768x768_20200923_model.prototxt'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/ssd_regnetx-1.6gf_fpn_bgr_lite_768x768_20200923_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':37.0}, model_shortlist=None, compact_name='ssd-lite-regNetX-1.6gf-fpn-bgr-mmdet-coco-768x768', shortlisted=False)
        ),
        'od-8070':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((416,416), (416,416), backend='cv2'),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 4,
                     'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolov3_d53_relu_416x416_20210117_model.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'694, 698, 702'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolov3_d53_relu_416x416_20210117_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':30.7}, model_shortlist=None, compact_name='yolov3-d53-relu-mmdet-coco-416x416', shortlisted=False)
        ),
        'od-8080':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((512,512), (512,512), backend='cv2'),
            session=onnx_session_type(**sessions.get_onnx_bgr_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 4,
                     'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolov3_regnetx-1.6gf_bgr_lite_512x512_20210202_model.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'823, 830, 837'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolov3_regnetx-1.6gf_bgr_lite_512x512_20210202_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':30.7}, model_shortlist=None, compact_name='yolov3-lite-regNetX-1.6gf-bgr-mmdet-coco-512x512', shortlisted=False)
        ),
        'od-8090':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((512,512), (512,512), backend='cv2'),
            session=onnx_session_type(**sessions.get_onnx_bgr_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 5,
                     'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/retinanet_regnetx-800mf_fpn_bgr_lite_512x512_20200908_model.prototxt'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/retinanet_regnetx-800mf_fpn_bgr_lite_512x512_20200908_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':33.0}, model_shortlist=None, compact_name='retina-lite-regNetX-800mf-fpn-bgr-mmdet-coco-512x512', shortlisted=False)
        ),
        # yolox lite versions from edgeai-mmdet
        'od-8200':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(416, 416, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolox_nano_lite_416x416_20220214_model.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '1033, 711, 712, 713, 727, 728, 729, 743, 744, 745'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolox_nano_lite_416x416_20220214_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 24.8}, model_shortlist=10, compact_name='yolox-nano-lite-mmdet-coco-416x416', shortlisted=True, recommended=True)
        ),         
        'od-8210':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(416, 416, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolox_tiny_lite_416x416_20220217_model.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '1033, 711, 712, 713, 727, 728, 728, 743, 744, 745'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolox_tiny_lite_416x416_20220217_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 30.5}, model_shortlist=20, compact_name='yolox-tiny-lite-mmdet-coco-416x416', shortlisted=True)
        ),
        # for some reason, the model export has an issue in only 640x640 resolution - the exported model doesn't have NMS OP
        # this causes issue in SoCs that do not use tidl_offload
        'od-8220':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolox_s_lite_640x640_20220221_model.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '1033, 711, 712, 713, 727, 728, 728, 743, 744, 745'},
                    fast_calibration=True),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolox_s_lite_640x640_20220221_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 38.3}, model_shortlist=10, compact_name='yolox-s-lite-mmdet-coco-640x640', shortlisted=True, recommended=True)
        ),
        'od-8230':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolox_m_lite_640x640_20220228_model.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '1256, 934, 935, 936, 950, 951, 952, 966, 967, 968'},
                    fast_calibration=True),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolox_m_lite_640x640_20220228_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 44.4}, model_shortlist=None, compact_name='yolox-m-lite-mmdet-coco-640x640', shortlisted=True)
        ),
        'od-8260':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(320, 320, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolox_femto_lite_320x320_20230407_model.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '1033, 711, 712, 713, 727, 728, 729, 743, 744, 745'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolox_femto_lite_320x320_20230407_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 12.7}, model_shortlist=None, compact_name='yolox-femto-lite-mmdet-coco-320x320', shortlisted=False)
        ),
        'od-8270':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(320, 320, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                   det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolox_pico_lite_320x320_20230410_model.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '1033, 711, 712, 713, 727, 728, 729, 743, 744, 745'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-mmdet/yolox_pico_lite_320x320_20230410_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 17.9}, model_shortlist=20, compact_name='yolox-pico-lite-mmdet-coco-320x320', shortlisted=True, recommended=True)
        ),
        #################################################################
        #       MXNET MODELS
        #################################################################
        # moved to experimental
        #################################################################
        #       TFLITE MODELS
        #################tflite models###################################
        # mlperf edge: detection - ssd_mobilenet_v1_coco_2018_01_28 expected_metric: 23.0% ap[0.5:0.95] accuracy
        'od-2000':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite((300,300), (300,300), backend='cv2'),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(det_options=True),
                model_path=f'{settings.models_path}/vision/detection/coco/mlperf/ssd_mobilenet_v1_coco_20180128.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':23.0}, model_shortlist=30, compact_name='ssd-mobV1-coco-mlperf-300x300', shortlisted=True)
        ),
        # mlperf mobile: detection - ssd_mobilenet_v2_coco_300x300 - expected_metric: 22.0% COCO AP[0.5-0.95]
        'od-2010':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite((300,300), (300,300), backend='cv2'),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(det_options=True),
                model_path=f'{settings.models_path}/vision/detection/coco/mlperf/ssd_mobilenet_v2_300_float.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':22.0}, model_shortlist=20, compact_name='ssd-mobV2-coco-mlperf-300x300', shortlisted=True)
        ),
        #################################################################
        # tensorflow1.0 models: detection - ssdlite_mobiledet_dsp_320x320_coco_2020_05_19 expected_metric: 28.9% ap[0.5:0.95] accuracy
        'od-2020':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite((320,320), (320,320), backend='cv2'),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(det_options=True),
                model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':28.9}, model_shortlist=10, compact_name='ssdLite-mobDet-DSP-coco-320x320', shortlisted=True, recommended=True)
        ),
        # tensorflow1.0 models: detection - ssdlite_mobiledet_edgetpu_320x320_coco_2020_05_19 expected_metric: 25.9% ap[0.5:0.95] accuracy
        'od-2030':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite((320,320), (320,320), backend='cv2'),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(det_options=True),
                model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssdlite_mobiledet_edgetpu_320x320_coco_20200519.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':25.9}, model_shortlist=40, compact_name='ssdLite-mobDet-EdgeTPU-coco-320x320', shortlisted=True)
        ),
        # tensorflow1.0 models: detection - ssdlite_mobilenet_v2_coco_2018_05_09 expected_metric: 22.0% ap[0.5:0.95] accuracy
        'od-2060':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite((300,300), (300,300), backend='cv2'),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(det_options=True),
                model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssdlite_mobilenet_v2_coco_20180509.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':22.0}, model_shortlist=None, compact_name='ssdlite-mobV2-coco-300x300', shortlisted=False)
        ),
        'od-2070':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite((640,640), (640,640), backend='cv2'),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(det_options=True),
                model_path=f'{settings.models_path}/vision/detection/coco/tf2-models/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':29.1}, model_shortlist=90, compact_name='ssd-mobV1-fpn-coco-tpu-8-640x640', shortlisted=False)
        ),
        'od-2080':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite((320,320), (320,320), backend='cv2'),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(det_options=True),
                model_path=f'{settings.models_path}/vision/detection/coco/tf2-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':22.2}, model_shortlist=30, compact_name='ssd-mobV2-fpnlite-coco-tpu-8-320x320', shortlisted=True)
        ),
        'od-2090':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite((640,640), (640,640), backend='cv2'),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(det_options=True),
                model_path=f'{settings.models_path}/vision/detection/coco/tf2-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':28.2}, model_shortlist=None, compact_name='ssd-mobV2-fpnlite-coco-tpu-8-640x640', shortlisted=False)
        ),
        'od-2100':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite((640,640), (640,640), backend='cv2'),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(det_options=True, fast_calibration=True),
                model_path=f'{settings.models_path}/vision/detection/coco/tf2-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':34.3}, model_shortlist=None, compact_name='ssd-res50V1-fpn-coco-tpu-8-640x640', shortlisted=False)
        ),
        'od-2110':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite((512,512), (512,512), backend='cv2'),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_optimization=False, input_mean=(123.675, 116.28, 103.53), input_scale=(0.01712475, 0.017507, 0.01742919)),
                runtime_options=settings.runtime_options_tflite_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 5,
                     'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/google-automl/efficientdet-lite0_bifpn_maxpool2x2_relu_ti-lite.prototxt'}),
                model_path=f'{settings.models_path}/vision/detection/coco/google-automl/efficientdet-lite0_bifpn_maxpool2x2_relu_ti-lite.tflite'),
            postprocess=postproc_detection_efficientdet_ti_lite_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90(label_offset=0)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':33.61}, model_shortlist=None, compact_name='efficientDet-ti-lite0-coco-512x512', shortlisted=False)
        ),
        # note although the name of the model said 320x320, the pipeline.config along with the model had 300x300
        # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
        'od-2130':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite((300,300), (300,300), backend='cv2'),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(det_options=True),
                model_path=f'{settings.models_path}/vision/detection/coco/tf2-models/ssd_mobilenet_v2_320x320_coco17_tpu-8.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':20.2}, model_shortlist=None, compact_name='ssd-mobV2-coco-tpu-8-320x320', shortlisted=False)
        ),
        'od-2150': utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite((384, 384), (384, 384), resize_with_pad=True, backend='cv2', pad_color=[127,127,127]),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_optimization=False, input_mean=(127.0,  127.0,  127.0), input_scale=(0.0078125, 0.0078125, 0.0078125)),
                runtime_options=settings.runtime_options_tflite_np2(
                   det_options=True, ext_options={'object_detection:meta_arch_type': 5,
                   'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/coco/google-automl/efficientdet_lite1_relu.prototxt',
                   'advanced_options:output_feature_16bit_names_list': 'efficientnet-lite1/stem/Relu;efficientnet-lite1/stem/tpu_batch_normalization/FusedBatchNormV3;efficientnet-lite1/blocks_0/tpu_batch_normalization/FusedBatchNormV3;efficientnet-lite1/blocks_0/depthwise_conv2d/depthwise;efficientnet-lite1/stem/conv2d/Conv2D, box_net/box-predict/BiasAdd;box_net/box-predict_4/separable_conv2d;box_net/box-predict/separable_conv2d;box_net/box-predict/bias, class_net/class-predict/BiasAdd;class_net/class-predict_4/separable_conv2d;class_net/class-predict/separable_conv2d;class_net/class-predict/bias, box_net/box-predict_1/BiasAdd;box_net/box-predict_4/separable_conv2d;box_net/box-predict_1/separable_conv2d;box_net/box-predict/bias, class_net/class-predict_1/BiasAdd;class_net/class-predict_4/separable_conv2d;class_net/class-predict_1/separable_conv2d;class_net/class-predict/bias, box_net/box-predict_2/BiasAdd;box_net/box-predict_4/separable_conv2d;box_net/box-predict_2/separable_conv2d;box_net/box-predict/bias, class_net/class-predict_2/BiasAdd;class_net/class-predict_4/separable_conv2d;class_net/class-predict_2/separable_conv2d;class_net/class-predict/bias, box_net/box-predict_3/BiasAdd;box_net/box-predict_4/separable_conv2d;box_net/box-predict_3/separable_conv2d;box_net/box-predict/bias, class_net/class-predict_3/BiasAdd;class_net/class-predict_4/separable_conv2d;class_net/class-predict_3/separable_conv2d;class_net/class-predict/bias, box_net/box-predict_4/BiasAdd;box_net/box-predict_4/separable_conv2d;box_net/box-predict/bias1, class_net/class-predict_4/BiasAdd;class_net/class-predict_4/separable_conv2d;class_net/class-predict/bias1'}),
                model_path=f'{settings.models_path}/vision/detection/coco/google-automl/efficientdet_lite1_relu.tflite'
                                        ),
            postprocess=postproc_transforms.get_transform_detection_tflite(normalized_detections=False, ignore_index=0,
                                                            formatter=postprocess.DetectionFormatting(dst_indices=(0,1,2,3,4,5), src_indices=(1,0,3,2,5,4)), resize_with_pad=True,),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90(label_offset=0)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 31.79}, model_shortlist=30, compact_name='efficientDet-lite1-relu-coco-384x384', shortlisted=False)
        ),
        'od-2170': utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite((512, 512), (512, 512), resize_with_pad=True, backend='cv2', pad_color=[127,127,127]),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_optimization=False, input_mean=(127.0,  127.0,  127.0), input_scale=(0.0078125, 0.0078125, 0.0078125)),
                runtime_options=settings.runtime_options_tflite_np2(
                   det_options=True, ext_options={'object_detection:meta_arch_type': 5,
                   'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/coco/google-automl/efficientdet_lite3_relu.prototxt',
                   'advanced_options:output_feature_16bit_names_list': 'efficientnet-lite3/stem/Relu;efficientnet-lite3/stem/tpu_batch_normalization/FusedBatchNormV3;efficientnet-lite3/blocks_0/tpu_batch_normalization/FusedBatchNormV3;efficientnet-lite3/blocks_0/depthwise_conv2d/depthwise;efficientnet-lite3/blocks_3/conv2d_1/Conv2D;efficientnet-lite3/stem/conv2d/Conv2D, box_net/box-predict/BiasAdd;box_net/box-predict_4/separable_conv2d;box_net/box-predict/separable_conv2d;box_net/box-predict/bias, class_net/class-predict/BiasAdd;class_net/class-predict_4/separable_conv2d;class_net/class-predict/separable_conv2d;class_net/class-predict/bias, box_net/box-predict_1/BiasAdd;box_net/box-predict_4/separable_conv2d;box_net/box-predict_1/separable_conv2d;box_net/box-predict/bias, class_net/class-predict_1/BiasAdd;class_net/class-predict_4/separable_conv2d;class_net/class-predict_1/separable_conv2d;class_net/class-predict/bias, box_net/box-predict_2/BiasAdd;box_net/box-predict_4/separable_conv2d;box_net/box-predict_2/separable_conv2d;box_net/box-predict/bias, class_net/class-predict_2/BiasAdd;class_net/class-predict_4/separable_conv2d;class_net/class-predict_2/separable_conv2d;class_net/class-predict/bias, box_net/box-predict_3/BiasAdd;box_net/box-predict_4/separable_conv2d;box_net/box-predict_3/separable_conv2d;box_net/box-predict/bias, class_net/class-predict_3/BiasAdd;class_net/class-predict_4/separable_conv2d;class_net/class-predict_3/separable_conv2d;class_net/class-predict/bias, box_net/box-predict_4/BiasAdd;box_net/box-predict_4/separable_conv2d;box_net/box-predict/bias1, class_net/class-predict_4/BiasAdd;class_net/class-predict_4/separable_conv2d;class_net/class-predict/bias1'}),
                model_path=f'{settings.models_path}/vision/detection/coco/google-automl/efficientdet_lite3_relu.tflite'),
            postprocess=postproc_transforms.get_transform_detection_tflite( normalized_detections=False, ignore_index=0,
                                                           formatter=postprocess.DetectionFormatting(dst_indices=(0,1,2,3,4,5), src_indices=(1,0,3,2,5,4)), resize_with_pad=True),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90(label_offset=0)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 38.33}, model_shortlist=None, compact_name='efficientDet-lite3-relu-coco-512x512', shortlisted=False)
        ),
        ###################################################################
        # complied for TVM - this model is repeated here and hard-coded to use tvmdlr session to generate an example tvmdlr artifact
        # mlperf edge: detection - ssd_mobilenet_v1_coco_2018_01_28 expected_metric: 23.0% ap[0.5:0.95] accuracy
        'od-5100':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite((300,300), (300,300), backend='cv2'),
            session=sessions.TVMDLRSession(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=settings.runtime_options_tflite_np2(det_options=True),
                model_path=f'{settings.models_path}/vision/detection/coco/mlperf/ssd_mobilenet_v1_coco_20180128.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':23.0}, model_shortlist=None, compact_name='ssd-mobV1-coco-mlperf-300x300', shortlisted=False)
        ),
        # tensorflow1.0 models: detection - ssdlite_mobiledet_dsp_320x320_coco_2020_05_19 expected_metric: 28.9% ap[0.5:0.95] accuracy
        'od-5120':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite((320,320), (320,320), backend='cv2'),
            session=sessions.TVMDLRSession(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=settings.runtime_options_tflite_np2(det_options=True),
                model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':28.9}, model_shortlist=10, compact_name='ssdLite-mobDet-DSP-coco-320x320', shortlisted=True, recommended=True)
        ),
    }
    return pipeline_configs

