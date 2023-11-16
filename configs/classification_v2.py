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
        'task_type': 'classification',
        'dataset_category': datasets.DATASET_CATEGORY_IMAGENET,
        'calibration_dataset': settings.dataset_cache['imagenet']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['imagenet']['input_dataset'],
        'postprocess': postproc_transforms.get_transform_classification()
    }

    quant_params_proto_path_disable_option = {constants.ADVANCED_OPTIONS_QUANT_FILE_KEY: ''}

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################jai-devkit models###############################
        # edgeai-torchvison: classification mobilenetv2_224x224 pytorch-qat expected_metric: 71.602% top-1 accuracy
        'cl-6700':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, with_onnxsim=True),
                runtime_options=settings.runtime_options_onnx_qat_v2(quantization_scale_type=constants.QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN, **quant_params_proto_path_disable_option),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/mobilenetv2_wt-v1_qat-v2-w8c-w8t_20230712_model.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':71.602}, model_shortlist=None)
        ),
        # edgeai-torchvison: classification mobilenetv2_224x224 pytorch-qat-sp2 expected_metric: 71.556% top-1 accuracy
        'cl-6710':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, with_onnxsim=True),
                runtime_options=settings.runtime_options_onnx_qat_v2(quantization_scale_type=constants.QUANTScaleType.QUANT_SCALE_TYPE_P2_QAT, **quant_params_proto_path_disable_option),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/mobilenetv2_wt-v1_qat-v2-w8csp2-w8tsp2_20230711_model.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':71.556}, model_shortlist=None)
        ),
        # edgeai-torchvison: classification resnet50_wt-v1_qat-w4c-w8t pytorch-qat-4bit-wt 224x224 expected_metric: 75.048% top-1 accuracy
        'cl-6720':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, with_onnxsim=True),
                runtime_options=settings.runtime_options_onnx_qat_v2(quantization_scale_type=constants.QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN, **quant_params_proto_path_disable_option),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/resnet50_wt-v1_qat-v2-w4c-w8t_20230713_model.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':75.048}, model_shortlist=None)
        ),
    }
    return pipeline_configs
