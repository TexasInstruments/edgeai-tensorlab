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

import cv2
from jai_benchmark import constants, utils, datasets, preprocess, sessions, postprocess, metrics


def get_configs(settings, work_dir):
    # get the sessions types to use for each model type
    onnx_session_type = settings.get_session_type(constants.MODEL_TYPE_ONNX)
    tflite_session_type = settings.get_session_type(constants.MODEL_TYPE_TFLITE)
    mxnet_session_type = settings.get_session_type(constants.MODEL_TYPE_MXNET)

    # for onnx and mxnet float models, we set non-power-of-2 scale for quant here - optional
    runtime_options_onnx = settings.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=False,
                                    runtime_options={'advanced_options:quantization_scale_type': 0})
    runtime_options_tflite = settings.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=False)
    runtime_options_mxnet = settings.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=False,
                                    runtime_options={'advanced_options:quantization_scale_type': 0})

    runtime_options_onnx_qat = settings.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=True)
    runtime_options_tflite_qat = settings.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=True)
    runtime_options_mxnet_qat = settings.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=True)

    # configs for each model pipeline
    cityscapes_cfg = {
        'task_type': 'segmentation',
        'calibration_dataset': settings.dataset_cache['cityscapes']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['cityscapes']['input_dataset'],
    }

    ade20k_cfg = {
        'task_type': 'segmentation',
        'calibration_dataset': settings.dataset_cache['ade20k']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['ade20k']['input_dataset'],
    }

    ade20k_cfg_class32 = {
        'task_type': 'segmentation',
        'calibration_dataset': settings.dataset_cache['ade20k32']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['ade20k32']['input_dataset'],
    }

    pascal_voc_cfg = {
        'task_type': 'segmentation',
        'calibration_dataset': settings.dataset_cache['voc2012']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['voc2012']['input_dataset'],
    }

    cocoseg21_cfg = {
        'task_type': 'segmentation',
        'calibration_dataset': settings.dataset_cache['cocoseg21']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['cocoseg21']['input_dataset'],
    }

    common_session_cfg = dict(work_dir=work_dir, target_device=settings.target_device)

    postproc_segmentation_onnx = settings.get_postproc_segmentation_onnx()
    postproc_segmenation_tflite = settings.get_postproc_segmentation_tflite(with_argmax=False)

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################mlperf models###################################
        #------------------------cityscapes models-----------------------
        # # edgeai: segmentation - deeplabv3lite_mobilenetv2_768x384_20190626-085932 expected_metric: 69.13% mean-iou
        # 'vseg-16-100-0':utils.dict_update(cityscapes_cfg,
        #     preprocess=settings.get_preproc_jai((384,768), (384,768), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx,
        #         model_path=f'{settings.models_path}/vision/segmentation/cityscapes/edgeai/deeplabv3lite_mobilenetv2_768x384_20190626_opset11.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':69.13})
        # ),
        # # edgeai: segmentation - fpnlite_aspp_mobilenetv2_768x384_20200120-135701 expected_metric: 70.48% mean-iou
        # 'vseg-16-101-0':utils.dict_update(cityscapes_cfg,
        #     preprocess=settings.get_preproc_jai((384,768), (384,768), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx,
        #         model_path=f'{settings.models_path}/vision/segmentation/cityscapes/edgeai/fpnlite_aspp_mobilenetv2_768x384_20200120_opset11.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':70.48})
        # ),
        # # edgeai: segmentation - unetlite_aspp_mobilenetv2_768x384_20200129-164340 expected_metric: 68.97% mean-iou
        # 'vseg-16-102-0':utils.dict_update(cityscapes_cfg,
        #     preprocess=settings.get_preproc_jai((384,768), (384,768), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx,
        #         model_path=f'{settings.models_path}/vision/segmentation/cityscapes/edgeai/unetlite_aspp_mobilenetv2_768x384_20200129_opset11.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':68.97})
        # ),
        # # edgeai: segmentation - fpnlite_aspp_regnetx800mf_768x384_20200911-144003 expected_metric: 72.01% mean-iou
        # 'vseg-16-103-0':utils.dict_update(cityscapes_cfg,
        #     preprocess=settings.get_preproc_jai((384,768), (384,768), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx,
        #         model_path=f'{settings.models_path}/vision/segmentation/cityscapes/edgeai/fpnlite_aspp_regnetx800mf_768x384_20200911_opset11.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':72.01})
        # ),
        # # edgeai: segmentation - fpnlite_aspp_regnetx1.6gf_1024x512_20200914-132016 expected_metric: 75.84% mean-iou
        # 'vseg-16-104-0':utils.dict_update(cityscapes_cfg,
        #     preprocess=settings.get_preproc_jai((512,1024), (512,1024), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx,
        #         model_path=f'{settings.models_path}/vision/segmentation/cityscapes/edgeai/fpnlite_aspp_regnetx1.6gf_1024x512_20200914_opset11.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':75.84})
        # ),
        # # edgeai: segmentation - fpnlite_aspp_regnetx3.2gf_1536x768_20200915-092738 expected_metric: 78.90% mean-iou
        # 'vseg-16-105-0':utils.dict_update(cityscapes_cfg,
        #     preprocess=settings.get_preproc_jai((768,1536), (768,1536), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx,
        #         model_path=f'{settings.models_path}/vision/segmentation/cityscapes/edgeai/fpnlite_aspp_regnetx3.2gf_1536x768_20200915_opset11.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':78.90})
        # ),
        # # torchvision: segmentation - torchvision deeplabv3-resnet50 - expected_metric: 73.5% MeanIoU.
        # 'vseg-16-300-0':utils.dict_update(cityscapes_cfg,
        #     preprocess=settings.get_preproc_onnx((520,1040), (520,1040), backend='cv2'),
        #     session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx,
        #         model_path=f'{settings.models_path}/vision/segmentation/cityscapes/torchvision/deeplabv3_resnet50_1040x520_20200901_opset11.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':73.5})
        # ),
        # # torchvision: segmentation - torchvision fcn-resnet50 - expected_metric: 71.6% MeanIoU.
        # 'vseg-16-301-0':utils.dict_update(cityscapes_cfg,
        #     preprocess=settings.get_preproc_onnx((520,1040), (520,1040), backend='cv2'),
        #     session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx,
        #         model_path=f'{settings.models_path}/vision/segmentation/cityscapes/torchvision/fcn_resnet50_1040x520_20200902_opset11.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':71.6})
        # ),
        #------------------------ade20k 32 class models-----------------------
        #  PTQ accuracy is good. Will remove in future.
        # 'vseg-18-100-8':utils.dict_update(ade20k_cfg_class32,
        #     preprocess=settings.get_preproc_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx_qat,
        #         model_path=f'{settings.models_path}/vision/segmentation/ade20k32/edgeai/deeplabv3lite_mobilenetv2_512x512_ade20k32_20210308_qat.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':51.61})
        # ),
        'vseg-18-101-8':utils.dict_update(ade20k_cfg_class32,
            preprocess=settings.get_preproc_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx_qat,
                model_path=f'{settings.models_path}/vision/segmentation/ade20k32/edgeai/unetlite_aspp_mobilenetv2_512x512_ade20k32_20210306_qat.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':49.98})
        ),
        #  PTQ accuracy is good. Will remove in future.
        # 'vseg-18-102-8':utils.dict_update(ade20k_cfg_class32,
        #     preprocess=settings.get_preproc_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx_qat,
        #         model_path=f'{settings.models_path}/vision/segmentation/ade20k32/edgeai/fpnlite_aspp_mobilenetv2_512x512_ade20k32_20210306_qat.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':50.93})
        # ),
        #  PTQ accuracy is good. Will remove in future.
        # 'vseg-18-103-8':utils.dict_update(ade20k_cfg_class32,
        #     preprocess=settings.get_preproc_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx_qat,
        #         model_path=f'{settings.models_path}/vision/segmentation/ade20k32/edgeai/fpnlite_aspp_mobilenetv2_1p4_512x512_ade20k32_20210307_qat.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':53.01})
        # ),
        #################################################################
        #       TFLITE MODELS
        #################mlperf models###################################
        # tensorflow-deeplab-cityscapes-segmentation- deeplabv3_mnv2_cityscapes_train - expected_metric: 73.57% MeanIoU.
        'vseg-16-400-0': utils.dict_update(cityscapes_cfg,
            preprocess=settings.get_preproc_tflite((1024, 2048), (1024, 2048), mean=(127.5, 127.5, 127.5), scale=(1/127.5, 1/127.5, 1/127.5), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite,
                model_path=f'{settings.models_path}/vision/segmentation/cityscapes/tf1-models/deeplabv3_mnv2_cityscapes_train_1024x2048.tflite'),
            postprocess=postproc_segmenation_tflite,
            model_info=dict(metric_reference={'accuracy_mean_iou%':73.57})
        ),
    }
    return pipeline_configs

