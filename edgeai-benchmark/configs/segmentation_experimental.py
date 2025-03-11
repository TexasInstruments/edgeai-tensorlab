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
from edgeai_benchmark import constants, utils, datasets, preprocess, sessions, postprocess, metrics


def get_configs(settings, work_dir):
    # get the sessions types to use for each model type
    onnx_session_type = settings.get_session_type(constants.MODEL_TYPE_ONNX)
    tflite_session_type = settings.get_session_type(constants.MODEL_TYPE_TFLITE)
    mxnet_session_type = settings.get_session_type(constants.MODEL_TYPE_MXNET)

    preproc_transforms = preprocess.PreProcessTransforms(settings)
    postproc_transforms = postprocess.PostProcessTransforms(settings)

    # configs for each model pipeline
    cityscapes_cfg = {
        'task_type': 'segmentation',
        'dataset_category': datasets.DATASET_CATEGORY_CITYSCAPES,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_CITYSCAPES]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_CITYSCAPES]['input_dataset'],
    }

    ade20k_cfg = {
        'task_type': 'segmentation',
        'dataset_category': datasets.DATASET_CATEGORY_ADE20K,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_ADE20K]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_ADE20K]['input_dataset'],
    }

    ade20k_cfg_class32 = {
        'task_type': 'segmentation',
        'dataset_category': datasets.DATASET_CATEGORY_ADE20K32,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_ADE20K32]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_ADE20K32]['input_dataset'],
    }

    pascal_voc_cfg = {
        'task_type': 'segmentation',
        'dataset_category': datasets.DATASET_CATEGORY_VOC2012,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_VOC2012]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_VOC2012]['input_dataset'],
    }

    cocoseg21_cfg = {
        'task_type': 'segmentation',
        'dataset_category': datasets.DATASET_CATEGORY_COCOSEG21,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_COCOSEG21]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_COCOSEG21]['input_dataset'],
    }

    postproc_segmentation_onnx = postproc_transforms.get_transform_segmentation_onnx()
    postproc_segmenation_tflite = postproc_transforms.get_transform_segmentation_tflite(with_argmax=False)

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################mlperf models###################################
        #------------------------robokit models-----------------------
        # 'ss-7610expt': utils.dict_update(robokitseg_cfg,
        #     preprocess=preproc_transforms.get_transform_jai((432,768), (432,768), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_np2(),
        #         model_path=f'{settings.models_path}/vision/segmentation/ti-robokit/edgeai-tv/deeplabv3plus_mnetv2_edgeailite_robokit_768x432.onnx'),
        #     postprocess=postproc_transforms.get_transform_segmentation_onnx(),
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':54.1})
        # ),
        #------------------------------------------------------------------
        # #the onnxrt compiled model config that was here was moved to segmentation.py
        #------------------------------------------------------------------
        # 'ss-5818expt': utils.dict_update(robokitseg_cfg,
        #     preprocess=preproc_transforms.get_transform_jai((432,768), (432,768), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=sessions.TVMDLRSession(**sessions.get_jai_quant_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_qat_v1(),
        #         model_path=f'{settings.models_path}/vision/segmentation/ti-robokit/edgeai-tv/deeplabv3plus_mnetv2_edgeailite_robokit_768x432_qat-p2.onnx'),
        #     postprocess=postproc_transforms.get_transform_segmentation_onnx(),
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':54.1}, model_shortlist=20)
        # ),
        #------------------------cityscapes models-----------------------
        # # edgeai: segmentation - deeplabv3lite_mobilenetv2_768x384_20190626-085932 expected_metric: 69.13% mean-iou
        # 'ss-8500expt':utils.dict_update(cityscapes_cfg,
        #     preprocess=preproc_transforms.get_transform_jai((384,768), (384,768), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_p2(),
        #         model_path=f'{settings.models_path}/vision/segmentation/cityscapes/edgeai-tv/deeplabv3plus_mobilenetv2_edgeailite_768x384_20190626.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':69.13})
        # ),
        # # edgeai: segmentation - fpnlite_aspp_mobilenetv2_768x384_20200120-135701 expected_metric: 70.48% mean-iou
        # 'ss-8520expt':utils.dict_update(cityscapes_cfg,
        #     preprocess=preproc_transforms.get_transform_jai((384,768), (384,768), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_p2(),
        #         model_path=f'{settings.models_path}/vision/segmentation/cityscapes/edgeai-tv/fpn_aspp_mobilenetv2_edgeailite_768x384_20200120.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':70.48})
        # ),
        # # edgeai: segmentation - unetlite_aspp_mobilenetv2_768x384_20200129-164340 expected_metric: 68.97% mean-iou
        # 'ss-8540expt':utils.dict_update(cityscapes_cfg,
        #     preprocess=preproc_transforms.get_transform_jai((384,768), (384,768), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_p2(),
        #         model_path=f'{settings.models_path}/vision/segmentation/cityscapes/edgeai-tv/unet_aspp_mobilenetv2_edgeailite_768x384_20200129.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':68.97})
        # ),
        # # edgeai: segmentation - fpnlite_aspp_regnetx800mf_768x384_20200911-144003 expected_metric: 72.01% mean-iou
        # 'ss-8560expt':utils.dict_update(cityscapes_cfg,
        #     preprocess=preproc_transforms.get_transform_jai((384,768), (384,768), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_p2(),
        #         model_path=f'{settings.models_path}/vision/segmentation/cityscapes/edgeai-tv/fpn_aspp_regnetx800mf_edgeailite_768x384_20200911.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':72.01})
        # ),
        # # edgeai: segmentation - fpnlite_aspp_regnetx1.6gf_1024x512_20200914-132016 expected_metric: 75.84% mean-iou
        # 'ss-8570expt':utils.dict_update(cityscapes_cfg,
        #     preprocess=preproc_transforms.get_transform_jai((512,1024), (512,1024), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_p2(),
        #         model_path=f'{settings.models_path}/vision/segmentation/cityscapes/edgeai-tv/fpn_aspp_regnetx1.6gf_edgeailite_1024x512_20200914.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':75.84})
        # ),
        # # edgeai: segmentation - fpnlite_aspp_regnetx3.2gf_1536x768_20200915-092738 expected_metric: 78.90% mean-iou
        # 'ss-8580expt':utils.dict_update(cityscapes_cfg,
        #     preprocess=preproc_transforms.get_transform_jai((768,1536), (768,1536), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_p2(),
        #         model_path=f'{settings.models_path}/vision/segmentation/cityscapes/edgeai-tv/fpn_aspp_regnetx3.2gf_edgeailite_1536x768_20200915.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':78.90})
        # ),
        # # torchvision: segmentation - torchvision deeplabv3-resnet50 - expected_metric: 73.5% MeanIoU.
        # 'ss-8590expt':utils.dict_update(cityscapes_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx((520,1040), (520,1040), backend='cv2'),
        #     session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_p2(),
        #         model_path=f'{settings.models_path}/vision/segmentation/cityscapes/torchvision/deeplabv3_resnet50_1040x520_20200901.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':73.5})
        # ),
        # # torchvision: segmentation - torchvision fcn-resnet50 - expected_metric: 71.6% MeanIoU.
        # 'ss-8600expt':utils.dict_update(cityscapes_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx((520,1040), (520,1040), backend='cv2'),
        #     session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_p2(),
        #         model_path=f'{settings.models_path}/vision/segmentation/cityscapes/torchvision/fcn_resnet50_1040x520_20200902.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':71.6})
        # ),
        #------------------------ade20k 32 class models-----------------------
        #  PTQ accuracy is good. Will remove in future.
        # 'ss-8618expt':utils.dict_update(ade20k_cfg_class32,
        #     preprocess=preproc_transforms.get_transform_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**sessions.get_jai_quant_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_qat_v1(),
        #         model_path=f'{settings.models_path}/vision/segmentation/ade20k32/edgeai-tv/deeplabv3plus_mobilenetv2_edgeailtie_512x512_20210308_qat.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':51.61})
        # ),
        # 'ss-8638expt':utils.dict_update(ade20k_cfg_class32,
        #     preprocess=preproc_transforms.get_transform_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**sessions.get_jai_quant_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_qat_v1(),
        #         model_path=f'{settings.models_path}/vision/segmentation/ade20k32/edgeai-tv/unetlite_aspp_mobilenetv2_512x512_ade20k32_20210306_qat.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':49.98})
        # ),
        #  PTQ accuracy is good. Will remove in future.
        # 'ss-8658expt':utils.dict_update(ade20k_cfg_class32,
        #     preprocess=preproc_transforms.get_transform_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**sessions.get_jai_quant_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_qat_v1(),
        #         model_path=f'{settings.models_path}/vision/segmentation/ade20k32/edgeai-tv/fpn_aspp_mobilenetv2_edgeailite_512x512_20210306_qat.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':50.93})
        # ),
        #  PTQ accuracy is good. Will remove in future.
        # 'ss-8678expt':utils.dict_update(ade20k_cfg_class32,
        #     preprocess=preproc_transforms.get_transform_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
        #     session=onnx_session_type(**sessions.get_jai_quant_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_onnx_qat_v1(),
        #         model_path=f'{settings.models_path}/vision/segmentation/ade20k32/edgeai-tv/fpn_aspp_mobilenetv2_1p4_edgeailite_512x512_20210307_qat.onnx'),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':53.01})
        # ),
        # #################################################################
        # #       MXNET MODELS
        # #################################################################
        # 'ss-5810expt':utils.dict_update(cocoseg21_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx((480,480), (480,480), backend='cv2'),
        #     session=mxnet_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_mxnet_np2(),
        #         model_path=[f'{settings.models_path}/vision/segmentation/cocoseg21/gluoncv-mxnet/fcn_resnet101_coco-symbol.json',
        #                     f'{settings.models_path}/vision/segmentation/cocoseg21/gluoncv-mxnet/fcn_resnet101_coco-0000.params'],
        #         model_type='mxnet', input_shape={'data':(1,3,480,480)}),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':None})
        # ),
        # 'ss-5820expt':utils.dict_update(cocoseg21_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx((480,480), (480,480), backend='cv2'),
        #     session=mxnet_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_mxnet_np2(),
        #         model_path=[f'{settings.models_path}/vision/segmentation/cocoseg21/gluoncv-mxnet/deeplab_resnet101_coco-symbol.json',
        #                     f'{settings.models_path}/vision/segmentation/cocoseg21/gluoncv-mxnet/deeplab_resnet101_coco-0000.params'],
        #         model_type='mxnet', input_shape={'data':(1,3,480,480)}),
        #     postprocess=postproc_segmentation_onnx,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':None})
        # ),
        # 'ss-5830expt':utils.dict_update(ade20k_cfg,
        #     preprocess=preproc_transforms.get_transform_mxnet((480,480), (480,480), backend='cv2', resize_with_pad=True),
        #     session=mxnet_session_type(**sessions.get_mxnet_session_cfg(settings, work_dir=work_dir),
        #         runtime_options=settings.runtime_options_mxnet_np2(),
        #         model_path=[f'{settings.models_path}/vision/segmentation/ade20k/gluoncv-mxnet/fcn_resnet50_ade-symbol.json',
        #                     f'{settings.models_path}/vision/segmentation/ade20k/gluoncv-mxnet/fcn_resnet50_ade-0000.params'],
        #         model_type='mxnet', input_shape={'data':(1,3,480,480)}),
        #     postprocess=postproc_segmentation_onnx,
        #     metric=dict(label_offset_target=-1),
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':39.5})
        # ),
        # #################################################################
        # #       TFLITE MODELS
        # #################mlperf models###################################
        # # tensorflow-deeplab-cityscapes-segmentation- deeplabv3_mnv2_cityscapes_train - expected_metric: 73.57% MeanIoU.
        # 'ss-2550expt': utils.dict_update(cityscapes_cfg,
        #     preprocess=preproc_transforms.get_transform_tflite((1024, 2048), (1024, 2048), backend='cv2'),
        #     session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_mean=(127.5, 127.5, 127.5), input_scale=(1/127.5, 1/127.5, 1/127.5)),
        #         runtime_options=settings.runtime_options_tflite_np2(),
        #         model_path=f'{settings.models_path}/vision/segmentation/cityscapes/tf1-models/deeplabv3_mnv2_cityscapes_train_1024x2048.tflite'),
        #     postprocess=postproc_segmenation_tflite,
        #     model_info=dict(metric_reference={'accuracy_mean_iou%':73.57})
        # ),
    }
    return pipeline_configs

