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
    common_cfg = {
        'task_type': 'classification',
        'calibration_dataset': settings.dataset_cache['imagenet']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['imagenet']['input_dataset'],
        'postprocess': settings.get_postproc_classification()
    }

    common_session_cfg = dict(work_dir=work_dir, target_device=settings.target_device)

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################jai-devkit models###############################
        # # torchvision: classification vgg16_224x224 expected_metric: 71.59% top-1 accuracy - too slow inference
        # 'vcls-10-306-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_onnx(),
        #     session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx,
        #         model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/vgg16_opset9.onnx'),
        #     model_info=dict(metric_reference={'accuracy_top1%':71.59})
        # ),
        #################github/onnx/models#############################
        # github onnx model: classification resnet18_v2 expected_metric: 69.70% top-1 accuracy
        'vcls-10-020-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx(),
            session=onnx_session_type(**common_session_cfg, runtime_options=runtime_options_onnx,
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/onnx-models/resnet18-v2-7.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':69.70})
        ),
        #################################################################
        #       MXNet MODELS
        #################################################################
        # mxnet : gluoncv model : classification - hrnet_w30_c - - reference accuracy: is from hrnet website, not from gluoncv
        'vcls-10-064-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx(backend='cv2'),
            session=mxnet_session_type(**common_session_cfg, runtime_options=runtime_options_mxnet_p2,
                model_path=[f'{settings.models_path}/vision/classification/imagenet1k/gluoncv-mxnet/hrnet_w30_c-symbol.json',
                            f'{settings.models_path}/vision/classification/imagenet1k/gluoncv-mxnet/hrnet_w30_c-0000.params'],
                model_type='mxnet', input_shape={'data':(1,3,224,224)}),
            model_info=dict(metric_reference={'accuracy_top1%':78.2})
        ),
        #################################################################
        #       TFLITE MODELS
        ##################tensorflow models##############################
        # tensorflow/models: classification mobilenetv2_224x224 quant expected_metric: 70.8% top-1 accuracy
        'vcls-10-401-8':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite(),
            session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite,
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v2_1.0_224_quant.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':70.8})
        ),
        # # tf hosted models: classification nasnet mobile expected_metric: 73.9% top-1 accuracy
        # 'vcls-10-408-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_tflite(),
        #     session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite,
        #         model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/nasnet_mobile.tflite'),
        #     metric=dict(label_offset_pred=-1),
        #     model_info=dict(metric_reference={'accuracy_top1%':73.9})
        # ),
        # # tensorflow/tpu: classification efficinetnet-lite2_260x260 expected_metric: 77.6% top-1 accuracy
        # 'vcls-10-432-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_tflite(297, 260),
        #     session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite,
        #         model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-lite2-fp32.tflite'),
        #     model_info=dict(metric_reference={'accuracy_top1%':77.6})
        # ),
        # ##################tf2-models#####################################################
        # # tf2_models: classification xception expected_metric: 79.0% top-1 accuracy
        # 'vcls-10-450-0':utils.dict_update(common_cfg,
        #     preprocess=settings.get_preproc_tflite(342, 299),
        #     session=tflite_session_type(**common_session_cfg, runtime_options=runtime_options_tflite,
        #         model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf2-models/xception.tflite'),
        #     model_info=dict(metric_reference={'accuracy_top1%':79.0})
        # ),
    }
    return pipeline_configs
