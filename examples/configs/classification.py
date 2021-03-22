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

from jacinto_ai_benchmark import constants, utils, datasets, preprocess, sessions, postprocess, metrics


def get_configs(settings, work_dir):
    # get the sessions types to use for each model type
    onnx_session_type = settings.get_session_type(constants.MODEL_TYPE_ONNX)
    tflite_session_type = settings.get_session_type(constants.MODEL_TYPE_TFLITE)
    mxnet_session_type = settings.get_session_type(constants.MODEL_TYPE_MXNET)

    # get the session cfgs to be used for float models
    onnx_session_cfg = settings.get_session_cfg(constants.MODEL_TYPE_ONNX, is_qat=False)
    tflite_session_cfg = settings.get_session_cfg(constants.MODEL_TYPE_TFLITE, is_qat=False)
    mxnet_session_cfg = settings.get_session_cfg(constants.MODEL_TYPE_MXNET, is_qat=False)

    # get the session cfgs to be used for qat models
    onnx_session_cfg_qat = settings.get_session_cfg(constants.MODEL_TYPE_ONNX, is_qat=True)
    tflite_session_cfg_qat = settings.get_session_cfg(constants.MODEL_TYPE_TFLITE, is_qat=True)
    mxnet_session_cfg_qat = settings.get_session_cfg(constants.MODEL_TYPE_MXNET, is_qat=True)

    # configs for each model pipeline
    common_cfg = {
        'pipeline_type': settings.pipeline_type,
        'task_type': 'classification',
        'verbose': settings.verbose,
        'run_import': settings.run_import,
        'run_inference': settings.run_inference,
        'calibration_dataset': settings.dataset_cache['imagenet']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['imagenet']['input_dataset'],
        'postprocess': settings.get_postproc_classification()
    }

    common_session_cfg = dict(work_dir=work_dir, target_device=settings.target_device)

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################jai-devkit models###############################
        # torchvision: classification mobilenetv2_224x224 expected_metric: 71.88% top-1 accuracy
        'vcls-10-302-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx(),
            session=onnx_session_type(**common_session_cfg, **onnx_session_cfg,
                model_path=f'{settings.modelzoo_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_opset9.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':71.88})
        ),
        #################################################################
        #       MXNet MODELS
        #################################################################
        # mxnet : gluoncv model : classification - mobilenetv2_1.0 - accuracy: 72.04% top1
        'vcls-10-060-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_onnx(backend='cv2'),
            session=mxnet_session_type(**common_session_cfg, **mxnet_session_cfg,
                model_path=[f'{settings.modelzoo_path}/vision/classification/imagenet1k/gluoncv-mxnet/mobilenetv2_1.0-symbol.json',
                            f'{settings.modelzoo_path}/vision/classification/imagenet1k/gluoncv-mxnet/mobilenetv2_1.0-0000.params'],
                model_type='mxnet', input_shape={'data':(1,3,224,224)}),
            model_info=dict(metric_reference={'accuracy_top1%':72.04})
        ),
        #################################################################
        #       TFLITE MODELS
        ##################tensorflow models##############################
        # mlperf/tf1 model: classification mobilenet_v1_224x224 expected_metric: 71.676 top-1 accuracy
        'vcls-10-010-0':utils.dict_update(common_cfg,
            preprocess=settings.get_preproc_tflite(),
            session=tflite_session_type(**common_session_cfg, **tflite_session_cfg,
                model_path=f'{settings.modelzoo_path}/vision/classification/imagenet1k/mlperf/mobilenet_v1_1.0_224.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':71.676})
        ),
    }
    return pipeline_configs
