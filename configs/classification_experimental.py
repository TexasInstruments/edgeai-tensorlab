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

from jai_benchmark import constants, utils, datasets, preprocess, sessions, postprocess, metrics


def get_configs(settings, work_dir):
    # get the sessions types to use for each model type
    onnx_session_type = settings.get_session_type(constants.MODEL_TYPE_ONNX)
    tflite_session_type = settings.get_session_type(constants.MODEL_TYPE_TFLITE)
    mxnet_session_type = settings.get_session_type(constants.MODEL_TYPE_MXNET)

    preproc_transforms = preprocess.PreProcessTransforms(settings)
    postproc_transforms = postprocess.PostProcessTransforms(settings)

    # configs for each model pipeline
    common_cfg = {
        'task_type': 'classification',
        'calibration_dataset': settings.dataset_cache['imagenet']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['imagenet']['input_dataset'],
        'postprocess': postproc_transforms.get_transform_classification()
    }

    common_session_cfg = sessions.get_common_session_cfg(settings, work_dir=work_dir)
    onnx_session_cfg = sessions.get_onnx_session_cfg(settings, work_dir=work_dir)
    onnx_bgr_session_cfg = sessions.get_onnx_bgr_session_cfg(settings, work_dir=work_dir)
    onnx_quant_session_cfg = sessions.get_onnx_quant_session_cfg(settings, work_dir=work_dir)
    onnx_bgr_quant_session_cfg = sessions.get_onnx_bgr_quant_session_cfg(settings, work_dir=work_dir)
    jai_session_cfg = sessions.get_jai_session_cfg(settings, work_dir=work_dir)
    jai_quant_session_cfg = sessions.get_jai_quant_session_cfg(settings, work_dir=work_dir)
    mxnet_session_cfg = sessions.get_mxnet_session_cfg(settings, work_dir=work_dir)
    tflite_session_cfg = sessions.get_tflite_session_cfg(settings, work_dir=work_dir)
    tflite_quant_session_cfg = sessions.get_tflite_quant_session_cfg(settings, work_dir=work_dir)

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################jai-devkit models###############################
        # # pycls: classification regnetx400mf_224x224 expected_metric: 72.7% top-1 accuracy
        # 'cl-6120':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx(reverse_channels=True),
        #     session=onnx_session_type(**onnx_bgr_session_cfg,
        #         runtime_options=settings.runtime_options_onnx_np2(),
        #         model_path=f'{settings.models_path}/vision/classification/imagenet1k/fbr-pycls/regnetx-400mf.onnx'),
        #     model_info=dict(metric_reference={'accuracy_top1%':72.7})
        # ),
        # # pycls: classification regnetx800mf_224x224 expected_metric: 75.2% top-1 accuracy
        # 'cl-6130':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx(reverse_channels=True),
        #     session=onnx_session_type(**onnx_bgr_session_cfg,
        #         runtime_options=settings.runtime_options_onnx_np2(),
        #         model_path=f'{settings.models_path}/vision/classification/imagenet1k/fbr-pycls/regnetx-800mf.onnx'),
        #     model_info=dict(metric_reference={'accuracy_top1%':75.2})
        # ),
        # # pycls: classification regnetx1.6gf_224x224 expected_metric: 77.0% top-1 accuracy
        # 'cl-6140':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx(reverse_channels=True),
        #     session=onnx_session_type(**onnx_bgr_session_cfg,
        #         runtime_options=settings.runtime_options_onnx_np2(),
        #         model_path=f'{settings.models_path}/vision/classification/imagenet1k/fbr-pycls/regnetx-1.6gf.onnx'),
        #     model_info=dict(metric_reference={'accuracy_top1%':77.0})
        # ),
        #
        # # torchvision: classification vgg16_224x224 expected_metric: 71.59% top-1 accuracy - too slow inference
        # 'cl-6370':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx(),
        #     session=onnx_session_type(**onnx_session_cfg, runtime_options=settings.runtime_options_onnx_np2(),
        #         model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/vgg16.onnx'),
        #     model_info=dict(metric_reference={'accuracy_top1%':71.59})
        # ),
        #
        # # jai-devkit: classification mobilenetv3_large_lite qat expected_metric: 71.614% top-1 accuracy
        # 'cl-6508':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx(),
        #     session=onnx_session_type(**onnx_quant_session_cfg,
        #         runtime_options=settings.runtime_options_onnx_qat(),
        #         model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv/mobilenet_v3_lite_large_qat-p2_20210507.onnx'),
        #     model_info=dict(metric_reference={'accuracy_top1%':71.614})
        # ),
        # # jai-devkit: classification mobilenetv3_large_lite_x2r expected_metric: 74.160% top-1 accuracy
        # 'cl-6510':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx(),
        #     session=onnx_session_type(**onnx_session_cfg,
        #         runtime_options=settings.runtime_options_onnx_p2(),
        #         model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv/mobilenet_v3_lite_large_x2r_20210522.onnx'),
        #     model_info=dict(metric_reference={'accuracy_top1%':74.160})
        # ),
        #
        #################github/onnx/models#############################
        # # github onnx model: classification resnet18_v2 expected_metric: 69.70% top-1 accuracy
        # 'cl-6000':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx(),
        #     session=onnx_session_type(**onnx_session_cfg,
        #         runtime_options=settings.runtime_options_onnx_np2(),
        #         model_path=f'{settings.models_path}/vision/classification/imagenet1k/onnx-models/resnet18-v2-7.onnx'),
        #     model_info=dict(metric_reference={'accuracy_top1%':69.70})
        # ),
        #
        # #################################################################
        # #       MXNet MODELS
        # #################################################################
        # # mxnet : gluoncv model : classification - hrnet_w30_c - - reference accuracy: is from hrnet website, not from gluoncv
        # 'cl-3510':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_onnx(backend='cv2'),
        #     session=mxnet_session_type(**onnx_session_cfg,
        #         runtime_options=settings.runtime_options_mxnet_np2(),
        #         model_path=[f'{settings.models_path}/vision/classification/imagenet1k/gluoncv-mxnet/hrnet_w30_c-symbol.json',
        #                     f'{settings.models_path}/vision/classification/imagenet1k/gluoncv-mxnet/hrnet_w30_c-0000.params'],
        #         model_type='mxnet', input_shape={'data':(1,3,224,224)}),
        #     model_info=dict(metric_reference={'accuracy_top1%':78.2})
        # ),
        #
        # #################################################################
        # #       TFLITE MODELS
        # ##################tensorflow models##############################
        # # tensorflow/models: classification mobilenetv2_224x224 quant expected_metric: 70.8% top-1 accuracy
        # 'cl-0018':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_tflite_quant(),
        #     session=tflite_session_type(**tflite_quant_session_cfg,
        #         runtime_options=settings.runtime_options_tflite_np2(),
        #         model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v2_1.0_224_quant.tflite'),
        #     metric=dict(label_offset_pred=-1),
        #     model_info=dict(metric_reference={'accuracy_top1%':70.8})
        # ),
        # # tf hosted models: classification nasnet mobile expected_metric: 73.9% top-1 accuracy
        # 'cl-0240':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_tflite(),
        #     session=tflite_session_type(**tflite_session_cfg,
        #         runtime_options=settings.runtime_options_tflite_np2(),
        #         model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/nasnet_mobile.tflite'),
        #     metric=dict(label_offset_pred=-1),
        #     model_info=dict(metric_reference={'accuracy_top1%':73.9})
        # ),
        # # tensorflow/tpu: classification efficinetnet-lite2_260x260 expected_metric: 77.6% top-1 accuracy
        # 'cl-0180':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.get_transform_tflite(297, 260),
        #     session=tflite_session_type(**tflite_session_cfg,
        #         runtime_options=settings.runtime_options_tflite_np2(),
        #         model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-lite2-fp32.tflite'),
        #     model_info=dict(metric_reference={'accuracy_top1%':77.6})
        # ),
        #
        # ##################tf2-models#####################################################
        # # tf2_models: classification xception expected_metric: 79.0% top-1 accuracy
        # 'cl-0250':utils.dict_update(common_cfg,
        #     preprocess=preproc_transforms.tflite_session_cfg(342, 299),
        #     session=tflite_session_type(**common_session_cfg,
        #         runtime_options=settings.runtime_options_tflite_np2(),
        #         model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf2-models/xception.tflite'),
        #     model_info=dict(metric_reference={'accuracy_top1%':79.0})
        # ),
    }
    return pipeline_configs
