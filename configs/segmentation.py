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

    # Default for ONNX/MXNET Models: Non-Power-2, TFLITE-Power2
    # For selected model we toggle based on which ever is better from accuracy perspective
    runtime_options_onnx_np2 = settings.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=False,
                                    runtime_options={'advanced_options:quantization_scale_type': 0})
    runtime_options_tflite_np2 = settings.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=False,
                                    runtime_options={'advanced_options:quantization_scale_type': 0})
    runtime_options_mxnet_np2 = settings.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=False,
                                    runtime_options={'advanced_options:quantization_scale_type': 0})

    runtime_options_onnx_p2 = settings.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=False,)
    runtime_options_tflite_p2 = settings.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=False,)
    runtime_options_mxnet_p2 = settings.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=False,)

    #This option should go away after testing
    use_default_power_2_setting = False
    if use_default_power_2_setting:
        runtime_options_onnx_p2 = runtime_options_onnx_np2
        runtime_options_tflite_np2 = runtime_options_tflite_p2
        runtime_options_mxnet_p2 = runtime_options_mxnet_np2

    runtime_options_onnx_qat = settings.get_runtime_options(constants.MODEL_TYPE_ONNX, is_qat=True)
    runtime_options_tflite_qat = settings.get_runtime_options(constants.MODEL_TYPE_TFLITE, is_qat=True)
    runtime_options_mxnet_qat = settings.get_runtime_options(constants.MODEL_TYPE_MXNET, is_qat=True)

    ade20k_cfg = {
        'task_type': 'segmentation',
        'calibration_dataset': settings.dataset_cache['ade20k']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['ade20k']['input_dataset'],
    }

    ade20k32_cfg = {
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
        # jai-pytorch: segmentation - fpnlite_aspp_regnetx400mf_ade20k32_384x384_20210314-205347 expected_metric: 51.03% mean-iou
        'vseg-18-110-0':utils.dict_update(ade20k32_cfg,
            preprocess=settings.get_preproc_jai((384,384), (384,384), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx_np2,
                model_path=f'{settings.models_path}/vision/segmentation/ade20k32/jai-pytorch/fpnlite_aspp_regnetx400mf_ade20k32_384x384_outby4.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':50.85})
        ),
        # jai-pytorch: segmentation - fpnlite_aspp_regnetx800mf_ade20k32_512x512_20210312-150048 expected_metric: 53.29% mean-iou
        'vseg-18-111-0':utils.dict_update(ade20k32_cfg,
            preprocess=settings.get_preproc_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx_p2,
                model_path=f'{settings.models_path}/vision/segmentation/ade20k32/jai-pytorch/fpnlite_aspp_regnetx800mf_ade20k32_512x512_outby4.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':53.16})
        ),
        ################# jacinto-ai ONNX models : ADE20k-Class32 ###################################
        'vseg-18-100-0':utils.dict_update(ade20k32_cfg,
            preprocess=settings.get_preproc_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx_np2,
                model_path=f'{settings.models_path}/vision/segmentation/ade20k32/jai-pytorch/deeplabv3lite_mobilenetv2_512x512_ade20k32_outby4.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':51.08})
        ),
        'vseg-18-101-0':utils.dict_update(ade20k32_cfg,
            preprocess=settings.get_preproc_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx_p2,
                model_path=f'{settings.models_path}/vision/segmentation/ade20k32/jai-pytorch/unetlite_aspp_mobilenetv2_512x512_ade20k32_outby2.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':50.07})
        ),
        'vseg-18-102-0':utils.dict_update(ade20k32_cfg,
            preprocess=settings.get_preproc_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx_np2,
                model_path=f'{settings.models_path}/vision/segmentation/ade20k32/jai-pytorch/fpnlite_aspp_mobilenetv2_512x512_ade20k32_outby4.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':50.55})
        ),
        'vseg-18-103-0':utils.dict_update(ade20k32_cfg,
            preprocess=settings.get_preproc_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_AREA),
            session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx_p2,
                model_path=f'{settings.models_path}/vision/segmentation/ade20k32/jai-pytorch/fpnlite_aspp_mobilenetv2_1p4_512x512_ade20k32_outby4.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':52.90})
        ),

        #------------------------coco 21 class-----------------------
        'vseg-21-100-0':utils.dict_update(cocoseg21_cfg,
            preprocess=settings.get_preproc_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_LINEAR),
            session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx_np2,
                model_path=f'{settings.models_path}/vision/segmentation/cocoseg21/jai-pytorch/deeplabv3lite_mobilenetv2_cocoseg21_512x512_20210327.onnx'),
            postprocess=postproc_segmentation_onnx,
            model_info=dict(metric_reference={'accuracy_mean_iou%':57.58})
        ),
        #################################################################
        #       TFLITE MODELS
        #################mlperf models###################################
        #mlperf: ade20k-segmentation (32 class) - deeplabv3_mnv2_ade20k_float - expected_metric??
        'vseg-18-010-0':utils.dict_update(ade20k32_cfg,
            preprocess=settings.get_preproc_tflite((512, 512), (512, 512), mean=(123.675, 116.28, 103.53), scale=(0.017125, 0.017507, 0.017429), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_np2,
                 model_path=f'{settings.models_path}/vision/segmentation/ade20k32/mlperf/deeplabv3_mnv2_ade20k32_float.tflite'),
            postprocess=postproc_segmenation_tflite,
            model_info=dict(metric_reference={'accuracy_mean_iou%':54.8})
        ),
        #################tensorflow models###################################
        #tensorflow-deeplab-ade20k-segmentation- deeplabv3_mnv2_ade20k_train_2018_12_03 - expected_metric: 32.04% MeanIoU.
        'vseg-17-400-0':utils.dict_update(ade20k_cfg,
            preprocess=settings.get_preproc_tflite((512, 512), (512, 512), mean=(123.675, 116.28, 103.53), scale=(0.017125, 0.017507, 0.017429), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_np2,
                 model_path=f'{settings.models_path}/vision/segmentation/ade20k/tf1-models/deeplabv3_mnv2_ade20k_train_2018_12_03_512x512.tflite'),
            postprocess=postproc_segmenation_tflite,
            model_info=dict(metric_reference={'accuracy_mean_iou%':32.04})
        ),
        # tensorflow-deeplab-pascal-voc-segmentation- deeplabv3_mnv2_dm05_pascal_trainaug - expected_metric: 70.19% MeanIoU.
        'vseg-19-400-0': utils.dict_update(pascal_voc_cfg, #pascalvoc2012 deeplab
            preprocess=settings.get_preproc_tflite((512, 512), (512, 512), mean=(127.5, 127.5, 127.5), scale=(1/127.5, 1/127.5, 1/127.5), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_p2,
                model_path=f'{settings.models_path}/vision/segmentation/voc2012/tf1-models/deeplabv3_mnv2_dm05_pascal_trainaug_512x512.tflite'),
            postprocess=postproc_segmenation_tflite,
            model_info=dict(metric_reference={'accuracy_mean_iou%':70.19})
       ),
        # tensorflow-deeplab-pascal-voc-segmentation- deeplabv3_mnv2_pascal_train_aug - expected_metric: 77.33% MeanIoU.
        'vseg-19-401-0': utils.dict_update(pascal_voc_cfg,  # pascalvoc2012 deeplab
            preprocess=settings.get_preproc_tflite((512, 512), (512, 512), mean=(127.5, 127.5, 127.5), scale=(1/127.5, 1/127.5, 1/127.5), backend='cv2'),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite_np2,
               model_path=f'{settings.models_path}/vision/segmentation/voc2012/tf1-models/deeplabv3_mnv2_pascal_trainaug_512x512.tflite'),
            postprocess=postproc_segmenation_tflite,
            model_info=dict(metric_reference={'accuracy_mean_iou%':77.33})
        ),
    }
    return pipeline_configs

