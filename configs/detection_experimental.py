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
import onnxruntime
ORT_DISABLE_ALL = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

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
    postproc_detection_mxnet = postproc_transforms.get_transform_detection_mxnet()

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################onnx models#####################################
        # edgeai-yolov5 models - IMPORTANT - see licence of the repository edgeai-yolov5 before using this model
        'od-8100expt':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640,  resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                     'object_detection:meta_layers_names_list':f'../edgeai-yolov5/pretrained_models/models/detection/coco/edgeai-yolov5-gplv3/yolov5s6_640_ti_lite_metaarch.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'370, 680, 990, 1300'},
                     fast_calibration=True),
                model_path=f'../edgeai-yolov5/pretrained_models/models/detection/coco/edgeai-yolov5-gplv3/yolov5s6_640_ti_lite_37p4_56p0.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':37.4}, model_shortlist=None)
        ),
        'od-8110expt':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(384, 384,  resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                     'object_detection:meta_layers_names_list':f'../edgeai-yolov5/pretrained_models/models/detection/coco/edgeai-yolov5-gplv3/yolov5s6_384_ti_lite_metaarch.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'168, 370, 680, 990, 1300'},
                     fast_calibration=True),
                model_path=f'../edgeai-yolov5/pretrained_models/models/detection/coco/edgeai-yolov5-gplv3/yolov5s6_384_ti_lite_32p8_51p2.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':32.8}, model_shortlist=None)
        ),
        'od-8120expt':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                     'object_detection:meta_layers_names_list':f'../edgeai-yolov5/pretrained_models/models/detection/coco/edgeai-yolov5-gplv3/yolov5m6_640_ti_lite_metaarch.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'228, 498, 808, 1118, 1428'},
                     fast_calibration=True),
                model_path=f'../edgeai-yolov5/pretrained_models/models/detection/coco/edgeai-yolov5-gplv3/yolov5m6_640_ti_lite_44p1_62p9.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False,  resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':44.1}, model_shortlist=None)
        ),
        'od-8130expt':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, resize_with_pad=True, backend='cv2', pad_color=[114,114,114]),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(0.003921568627, 0.003921568627, 0.003921568627)),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                     'object_detection:meta_layers_names_list':f'../edgeai-yolov5/pretrained_models/models/detection/coco/edgeai-yolov5-gplv3/yolov5l6_640_ti_lite_metaarch.prototxt',
                     'advanced_options:output_feature_16bit_names_list':'288, 626, 936, 1246, 1556'},
                     fast_calibration=True),
                model_path=f'../edgeai-yolov5/pretrained_models/models/detection/coco/edgeai-yolov5-gplv3/yolov5l6_640_ti_lite_47p1_65p6.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False,  resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()), #TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':47.1}, model_shortlist=None)
        ),
        #DETR mmdetection
        'od-8960expt':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx((800, 800), (800, 800), resize_with_pad=False, backend='cv2'),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={
                     'object_detection:meta_arch_type': 6,
                     'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL
                     }, fast_calibration=True),
                model_path=f'../edgeai-modelzoo/models/vision/detection/coco/edgeai-mmdet/detr_r50_8xb2-150e_20240722_model.onnx'),
            postprocess=postproc_transforms.get_transform_detection_mmdet_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=False, formatter=postprocess.DetectionBoxSL2BoxLS()),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 39.9}, model_shortlist=None, compact_name='DETR-r50-mmdet-transformer-coco-800x800', shortlisted=False)
        ),
        # edgeai-yolox models
        'od-8140expt': utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/coco/edgeai-yolox/yolox_s_ti_lite_metaarch.prototxt',
                    'advanced_options:output_feature_16bit_names_list': '471, 709, 843, 977'}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-yolox/yolox-s-ti-lite_39p1_57p9.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),  # TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 39.1})
        ),
        'od-8150expt': utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(640, 640, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/coco/edgeai-yolox/yolox_m_ti_lite_metaarch.prototxt',
                    'advanced_options:output_feature_16bit_names_list': "615, 932, 1066, 1200"}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-yolox/yolox_m_ti_lite_45p5_64p2.onnx'),
                postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),   # TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 45.5})
            ),
        'od-8180expt': utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(416, 416, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/coco/edgeai-yolox/yolox_tiny_ti_lite_metaarch.prototxt',
                    'advanced_options:output_feature_16bit_names_list': "471, 709, 843, 977"}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-yolox/yolox_tiny_ti_lite_32p0_49p5.onnx'),
                postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),    # TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 32.0})
            ),
        'od-8190expt': utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(416, 416, reverse_channels=True, resize_with_pad=[True, "corner"], backend='cv2', pad_color=[114, 114, 114]),
            session=onnx_session_type(**sessions.get_common_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(
                    det_options=True, ext_options={'object_detection:meta_arch_type': 6,
                    'object_detection:meta_layers_names_list': f'{settings.models_path}/vision/detection/coco/edgeai-yolox/yolox_nano_ti_lite_metaarch.prototxt',
                    'advanced_options:output_feature_16bit_names_list': "471, 709, 843, 977"}),
                model_path=f'{settings.models_path}/vision/detection/coco/edgeai-yolox/yolox_nano_ti_lite_26p1_41p8.onnx'),
            postprocess=postproc_transforms.get_transform_detection_yolov5_onnx( squeeze_axis=None, normalized_detections=False, resize_with_pad=True, formatter=postprocess.DetectionBoxSL2BoxLS()),
            # TODO: check this
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90(label_offset=1)),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%': 25.8})
        ),
        # edgeai-torchvision models
        # 'od-8160expt':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx((512,512), (512,512), backend='cv2'),
        #     session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_np2(
        #            det_options=True, ext_options={'object_detection:meta_arch_type': 3,
        #            'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-tv/ssdlite_mobilenet_v2_fpn_lite_512x512_20211015_dummypp.prototxt'}),
        #         model_path=f'{settings.models_path}/vision/detection/coco/edgeai-tv/ssdlite_mobilenet_v2_fpn_lite_512x512_20211015_dummypp.onnx'),
        #     postprocess=postproc_transforms.get_transform_detection_tv_onnx(),
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90(label_offset=0)),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':25.0})
        # ),
        # 'od-8170expt':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx((512,512), (512,512), backend='cv2'),
        #     session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_np2(
        #             det_options=True, ext_options={'object_detection:meta_arch_type': 3,
        #             'object_detection:meta_layers_names_list':f'{settings.models_path}/vision/detection/coco/edgeai-tv/ssdlite_regnet_x_800mf_fpn_lite_20211030_dummypp.prototxt'}),
        #         model_path=f'{settings.models_path}/vision/detection/coco/edgeai-tv/ssdlite_regnet_x_800mf_fpn_lite_20211030_dummypp.onnx'),
        #     postprocess=postproc_transforms.get_transform_detection_tv_onnx(),
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90(label_offset=0)),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':30.3})
        # ),
        # # yolov3: detection - yolov3 416x416 - expected_metric: 31.0% COCO AP[0.5-0.95]
        # 'od-8010expt':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx((416,416), (416,416), backend='cv2'),
        #     session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_mean=(0.0, 0.0, 0.0), input_scale=(1/255.0, 1/255.0, 1/255.0)),
        #         runtime_options=settings.runtime_options_onnx_p2(det_options=True),
        #         model_path=f'{settings.models_path}/vision/detection/coco/onnx-models/yolov3-10.onnx',
        #         input_shape=dict(input_1=(1,3,416,416), image_shape=(1,2)),
        #         extra_inputs=dict(image_shape=np.array([416,416], dtype=np.float32)[np.newaxis,...])),
        #     postprocess=postproc_detection_onnx,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':31.0})
        # ),
        #################################################################
        #       MXNET MODELS
        #################################################################
        # # mxnet : gluoncv model : detection - yolo3_darknet53_coco - accuracy: 36.0% ap[0.5:0.95], 57.2% ap50
        # 'od-5050expt':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx((416,416), (416,416), backend='cv2'),
        #     session=mxnet_session_type(**sessions.get_mxnet_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_mxnet_p2(),
        #         model_path=[f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/yolo3_darknet53_coco-symbol.json',
        #                     f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/yolo3_darknet53_coco-0000.params'],
        #         model_type='mxnet', input_shape={'data':(1,3,416,416)}),
        #     postprocess=postproc_detection_mxnet,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':36.0})
        # ),
        # # mxnet : gluoncv model : detection - center_net_resnet18_v1b_coco - accuracy: 26.6% ap[0.5:0.95], 28.1% ap50
        # 'od-5060expt':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx((512,512), (512,512), backend='cv2'),
        #     session=mxnet_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_mxnet_p2(),
        #         model_path=[f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/center_net_resnet18_v1b_coco-symbol.json',
        #                     f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/center_net_resnet18_v1b_coco-0000.params'],
        #         model_type='mxnet', input_shape={'data':(1,3,512,512)}),
        #     postprocess=postproc_detection_mxnet,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':26.6})
        # ),
        # # mxnet : gluoncv model : detection - yolo3_mobilenet1.0_coco - accuracy: 28.6% ap[0.5:0.95], 48.9% ap50
        # 'od-5020expt':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx((416,416), (416,416), backend='cv2'),
        #     session=mxnet_session_type(**sessions.get_mxnet_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_mxnet_np2(),
        #         model_path=[f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/yolo3_mobilenet1.0_coco-symbol.json',
        #                     f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/yolo3_mobilenet1.0_coco-0000.params'],
        #         model_type='mxnet', input_shape={'data':(1,3,416,416)}),
        #     postprocess=postproc_detection_mxnet,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':28.6})
        # ),
        # # mxnet : gluoncv model : detection - ssd_512_resnet50_v1_coco - accuracy: 30.6% ap[0.5:0.95], 50.0% ap50
        # 'od-5030expt':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx((512,512), (512,512), backend='cv2'),
        #     session=mxnet_session_type(**sessions.get_mxnet_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_mxnet_p2(),
        #         model_path=[f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_resnet50_v1_coco-symbol.json',
        #                     f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_resnet50_v1_coco-0000.params'],
        #         model_type='mxnet', input_shape={'data':(1,3,512,512)}),
        #     postprocess=postproc_detection_mxnet,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':30.6})
        # ),
        # # mxnet : gluoncv model : detection - ssd_512_mobilenet1.0_coco - accuracy: 21.7% ap[0.5:0.95], 39.2% ap50
        # 'od-5040expt':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx((512,512), (512,512), backend='cv2'),
        #     session=mxnet_session_type(**sessions.get_mxnet_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_mxnet_np2(),
        #         model_path=[f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_mobilenet1.0_coco-symbol.json',
        #                     f'{settings.models_path}/vision/detection/coco/gluoncv-mxnet/ssd_512_mobilenet1.0_coco-0000.params'],
        #         model_type='mxnet', input_shape={'data':(1,3,512,512)}),
        #     postprocess=postproc_detection_mxnet,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_80to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':21.7})
        # ),
        #################################################################
        #       TFLITE MODELS
        #################tflite models###################################
        'od-2040expt':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite((320,320), (320,320), backend='cv2'),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(detection_options=True),
                model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssd_mobilenet_v2_mnasfpn_shared_box_predictor_320x320_coco_sync_20200518.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':26.6})
        ),
        'od-2050expt':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite((640,640), (640,640), backend='cv2'),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(detection_options=True),
                model_path=f'{settings.models_path}/vision/detection/coco/tf1-models/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_20180703.tflite'),
            postprocess=postproc_detection_tflite,
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':32.0})
        ),
        # 'od-2120expt':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_tflite((1024,1024), (1024,1024), backend='cv2'),
        #     session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_tflite_np2(det_options=True),
        #         model_path=f'{settings.models_path}/vision/detection/coco/tf2-models/ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8.tflite'),
        #     postprocess=postproc_detection_tflite,
        #     metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
        #     model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':38.3})
        # ),
    }
    return pipeline_configs

