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

    ade20k_cfg = {
        'task_type': 'segmentation',
        'dataset_category': datasets.DATASET_CATEGORY_ADE20K,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_ADE20K]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_ADE20K]['input_dataset'],
    }

    ade20k32_cfg = {
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

    robokitseg_cfg = {
        'task_type': 'segmentation',
        'dataset_category': datasets.DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_TI_ROBOKIT_SEMSEG_ZED1HD]['input_dataset'],
    }

    postproc_segmentation_onnx = postproc_transforms.get_transform_segmentation_onnx()
    postproc_segmenation_tflite = postproc_transforms.get_transform_segmentation_tflite(with_argmax=False)

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################mlperf models###################################
        # edgeai: segmentation - fpnlite_aspp_regnetx400mf_ade20k32_384x384_20210314-205347 expected_metric: 51.03% mean-iou
        'ss-8690':utils.dict_update(ade20k32_cfg,
            preprocess=preproc_transforms.get_transform_jai((384,384), (384,384), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/segmentation/ade20k32/edgeai-tv/fpn_aspp_regnetx400mf_edgeailite_384x384_20210314_outby4.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':50.85}, model_shortlist=90)
        ),
        # edgeai: segmentation - fpnlite_aspp_regnetx800mf_ade20k32_512x512_20210312-150048 expected_metric: 53.29% mean-iou
        'ss-8700':utils.dict_update(ade20k32_cfg,
            preprocess=preproc_transforms.get_transform_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/segmentation/ade20k32/edgeai-tv/fpn_aspp_regnetx800mf_edgeailite_512x512_20210312_outby4.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':53.16}, model_shortlist=90)
        ),
        ################# jacinto-ai ONNX models : ADE20k-Class32 ###################################
        'ss-8610':utils.dict_update(ade20k32_cfg,
            preprocess=preproc_transforms.get_transform_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/segmentation/ade20k32/edgeai-tv/deeplabv3plus_mobilenetv2_edgeailite_512x512_20210308_outby4.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':51.08}, model_shortlist=10)
        ),
        'ss-8630':utils.dict_update(ade20k32_cfg,
            preprocess=preproc_transforms.get_transform_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/segmentation/ade20k32/edgeai-tv/unet_aspp_mobilenetv2_edgeailite_512x512_20210306_outby2.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':50.07}, model_shortlist=20)
        ),
        'ss-8650':utils.dict_update(ade20k32_cfg,
            preprocess=preproc_transforms.get_transform_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/segmentation/ade20k32/edgeai-tv/fpn_aspp_mobilenetv2_edgeailite_512x512_20210306_outby4.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':50.55}, model_shortlist=90)
        ),
        'ss-8670':utils.dict_update(ade20k32_cfg,
            preprocess=preproc_transforms.get_transform_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/segmentation/ade20k32/edgeai-tv/fpn_aspp_mobilenetv2_1p4_edgeailite_512x512_20210307_outby4.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':52.90}, model_shortlist=90)
        ),

        #------------------------coco 21 class-----------------------
        'ss-8710':utils.dict_update(cocoseg21_cfg,
            preprocess=preproc_transforms.get_transform_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_LINEAR),
            session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/segmentation/cocoseg21/edgeai-tv/deeplabv3plus_mobilenetv2_edgeailite_512x512_20210405.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':57.77}, model_shortlist=20)
        ),
        'ss-8720':utils.dict_update(cocoseg21_cfg,
            preprocess=preproc_transforms.get_transform_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_LINEAR),
            session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/segmentation/cocoseg21/edgeai-tv/fpn_aspp_regnetx800mf_edgeailite_512x512_20210405.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':61.09}, model_shortlist=20)
        ),
        'ss-8730':utils.dict_update(cocoseg21_cfg,
            preprocess=preproc_transforms.get_transform_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_LINEAR),
            session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/segmentation/cocoseg21/edgeai-tv/deeplabv3_mobilenet_v3_large_lite_512x512_20210527.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':60.80}, model_shortlist=90)
        ),
        'ss-8740':utils.dict_update(cocoseg21_cfg,
            preprocess=preproc_transforms.get_transform_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_LINEAR),
            session=onnx_session_type(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/segmentation/cocoseg21/edgeai-tv/lraspp_mobilenet_v3_large_lite_512x512_20210527.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':59.80}, model_shortlist=40)
        ),
        ###############huggingface transformer models######################
        'ss-8750':utils.dict_update(ade20k_cfg,
            preprocess=preproc_transforms.get_transform_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/segmentation/ade20k/hf-transformers/segformer_b0_finetuned_ade_512_512_transformers_simp.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':None}, model_shortlist=80)
        ),
        'ss-8760':utils.dict_update(ade20k_cfg,
            preprocess=preproc_transforms.get_transform_jai((640,640), (640,640), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/segmentation/ade20k/hf-transformers/segformer_b5_finetuned_ade_640_640_transformers_simp.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':None}, model_shortlist=80)
        ),
        ###############robokit segmentation model######################
        'ss-7618': utils.dict_update(robokitseg_cfg,
            preprocess=preproc_transforms.get_transform_jai((432,768), (432,768), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**sessions.get_jai_quant_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_qat_v1(),
                model_path=f'{settings.models_path}/vision/segmentation/ti-robokit/edgeai-tv/deeplabv3plus_mnetv2_edgeailite_robokit_768x432_qat-p2.onnx'),
            postprocess=postproc_transforms.get_transform_segmentation_onnx(),
            model_info=dict(metric_reference={'accuracy_mean_iou%':54.1}, model_shortlist=10)
        ),
        #################################################################
        #       MXNET MODELS
        #################################################################
        # TODO: add models. There are no mxnet segmentation models here, right now
        #################################################################
        #       TFLITE MODELS
        #################mlperf models###################################
        #mlperf: ade20k-segmentation (32 class) - deeplabv3_mnv2_ade20k_float - expected_metric??
        'ss-2580':utils.dict_update(ade20k32_cfg,
            preprocess=preproc_transforms.get_transform_tflite((512, 512), (512, 512), backend='cv2'),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_mean=(123.675, 116.28, 103.53), input_scale=(0.017125, 0.017507, 0.017429)),
                 runtime_options=settings.runtime_options_tflite_p2(),
                 model_path=f'{settings.models_path}/vision/segmentation/ade20k32/mlperf/deeplabv3_mnv2_ade20k32_float.tflite'),
            postprocess=postproc_segmenation_tflite,
            model_info=dict(metric_reference={'accuracy_mean_iou%':54.8}, model_shortlist=10)
        ),
        #################tensorflow models###################################
        #tensorflow-deeplab-ade20k-segmentation- deeplabv3_mnv2_ade20k_train_2018_12_03 - expected_metric: 32.04% MeanIoU.
        'ss-2540':utils.dict_update(ade20k_cfg,
            preprocess=preproc_transforms.get_transform_tflite((512, 512), (512, 512), backend='cv2'),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_mean=(123.675, 116.28, 103.53), input_scale=(0.017125, 0.017507, 0.017429)),
                 runtime_options=settings.runtime_options_tflite_np2(),
                 model_path=f'{settings.models_path}/vision/segmentation/ade20k/tf1-models/deeplabv3_mnv2_ade20k_train_20181203_512x512.tflite'),
            postprocess=postproc_segmenation_tflite,
            model_info=dict(metric_reference={'accuracy_mean_iou%':32.04}, model_shortlist=None)
        ),
        # tensorflow-deeplab-pascal-voc-segmentation- deeplabv3_mnv2_dm05_pascal_trainaug - expected_metric: 70.19% MeanIoU.
        'ss-2590': utils.dict_update(pascal_voc_cfg, #pascalvoc2012 deeplab
            preprocess=preproc_transforms.get_transform_tflite((512, 512), (512, 512), backend='cv2'),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_mean=(127.5, 127.5, 127.5), input_scale=(1/127.5, 1/127.5, 1/127.5)),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/segmentation/voc2012/tf1-models/deeplabv3_mnv2_dm05_pascal_trainaug_512x512.tflite'),
            postprocess=postproc_segmenation_tflite,
            model_info=dict(metric_reference={'accuracy_mean_iou%':70.19}, model_shortlist=None)
       ),
        # tensorflow-deeplab-pascal-voc-segmentation- deeplabv3_mnv2_pascal_train_aug - expected_metric: 77.33% MeanIoU.
        'ss-2600': utils.dict_update(pascal_voc_cfg,  # pascalvoc2012 deeplab
            preprocess=preproc_transforms.get_transform_tflite((512, 512), (512, 512), backend='cv2'),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_mean=(127.5, 127.5, 127.5), input_scale=(1/127.5, 1/127.5, 1/127.5)),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/segmentation/voc2012/tf1-models/deeplabv3_mnv2_pascal_trainaug_512x512.tflite'),
            postprocess=postproc_segmenation_tflite,
            model_info=dict(metric_reference={'accuracy_mean_iou%':77.33}, model_shortlist=None)
        ),
        ###################################################################
        # complied for TVM - this model is repeated here and hard-coded to use tvmdlr session to generate an example tvmdlr artifact
        'ss-5710':utils.dict_update(cocoseg21_cfg,
            preprocess=preproc_transforms.get_transform_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_LINEAR),
            session=sessions.TVMDLRSession(**sessions.get_jai_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/segmentation/cocoseg21/edgeai-tv/deeplabv3plus_mobilenetv2_edgeailite_512x512_20210405.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':57.77}, model_shortlist=10)
        ),
    }
    return pipeline_configs

