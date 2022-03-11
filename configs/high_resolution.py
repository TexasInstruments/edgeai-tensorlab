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

    # overriding with a small number as these take too much time.
    calibration_frames = 10
    num_frames = 100

    # configs for each model pipeline
    common_cfg = {
        'task_type': 'classification',
        'calibration_dataset': settings.dataset_cache['imagenet']['calibration_dataset'],
        'input_dataset': settings.dataset_cache['imagenet']['input_dataset'],
        'postprocess': postproc_transforms.get_transform_classification(),
        'calibration_frames': calibration_frames,
        'num_frames': num_frames
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

    hr_input_sizes = (512,1024)
    hr_input_sizes_x = [f'{s}x{s}' for s in hr_input_sizes]
    # use a fast calibration setting as the goal of these modles are performance estimation and not accuracy
    hr_runtime_options = {
        'accuracy_level': 0,
        'advanced_options:calibration_frames': calibration_frames,
        'advanced_options:calibration_iterations': 1,
        'advanced_options:high_resolution_optimization': 1
    }

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################jai-devkit models###############################
         # jai-devkit: classification mobilenetv1_224x224 expected_metric: 71.82% top-1 accuracy
        'cl-6061':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=hr_input_sizes[0], crop=hr_input_sizes[0]),
            session=onnx_session_type(**onnx_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), hr_runtime_options),
                model_path=f'{settings.models_path}/vision/high_resolution/imagenet1k/edgeai-tv/mobilenet_v1_20190906_{hr_input_sizes_x[0]}.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':None})
        ),
        'cl-6062':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=hr_input_sizes[1], crop=hr_input_sizes[1]),
            session=onnx_session_type(**onnx_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), hr_runtime_options),
                model_path=f'{settings.models_path}/vision/high_resolution/imagenet1k/edgeai-tv/mobilenet_v1_20190906_{hr_input_sizes_x[1]}.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':None})
        ),
        # jai-devkit: classification mobilenetv2_224x224 expected_metric: 72.13% top-1 accuracy
        'cl-6071':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=hr_input_sizes[0], crop=hr_input_sizes[0]),
            session=onnx_session_type(**onnx_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(), hr_runtime_options),
                model_path=f'{settings.models_path}/vision/high_resolution/imagenet1k/edgeai-tv/mobilenet_v2_20191224_{hr_input_sizes_x[0]}.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':None})
        ),
        'cl-6072':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=hr_input_sizes[1], crop=hr_input_sizes[1]),
            session=onnx_session_type(**onnx_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(), hr_runtime_options),
                model_path=f'{settings.models_path}/vision/high_resolution/imagenet1k/edgeai-tv/mobilenet_v2_20191224_{hr_input_sizes_x[1]}.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':None})
        ),
        # jai-devkit: classification mobilenetv2_1p4_224x224 expected_metric: 75.22% top-1 accuracy, QAT: 75.22%
        'cl-6151':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=hr_input_sizes[0], crop=hr_input_sizes[0]),
            session=onnx_session_type(**onnx_quant_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_qat(), hr_runtime_options),
                model_path=f'{settings.models_path}/vision/high_resolution/imagenet1k/edgeai-tv/mobilenet_v2_1p4_qat-p2_20210112_{hr_input_sizes_x[0]}.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':None})
        ),
        'cl-6152':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=hr_input_sizes[1], crop=hr_input_sizes[1]),
            session=onnx_session_type(**onnx_quant_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_qat(), hr_runtime_options),
                model_path=f'{settings.models_path}/vision/high_resolution/imagenet1k/edgeai-tv/mobilenet_v2_1p4_qat-p2_20210112_{hr_input_sizes_x[1]}.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':None})
        ),
        # torchvision: classification resnet18_224x224 expected_metric: 69.76% top-1 accuracy
        'cl-6101':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=hr_input_sizes[0], crop=hr_input_sizes[0]),
            session=onnx_session_type(**onnx_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), hr_runtime_options),
                model_path=f'{settings.models_path}/vision/high_resolution/imagenet1k/torchvision/resnet18_{hr_input_sizes_x[0]}.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':None})
        ),
        'cl-6102':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=hr_input_sizes[1], crop=hr_input_sizes[1]),
            session=onnx_session_type(**onnx_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), hr_runtime_options),
                model_path=f'{settings.models_path}/vision/high_resolution/imagenet1k/torchvision/resnet18_{hr_input_sizes_x[1]}.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':None})
        ),
        # torchvision: classification resnet50_224x224 expected_metric: 76.15% top-1 accuracy
        'cl-6111':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=hr_input_sizes[0], crop=hr_input_sizes[0]),
            session=onnx_session_type(**onnx_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(), hr_runtime_options),
                model_path=f'{settings.models_path}/vision/high_resolution/imagenet1k/torchvision/resnet50_{hr_input_sizes_x[0]}.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15})
        ),
        'cl-6112':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=hr_input_sizes[1], crop=hr_input_sizes[1]),
            session=onnx_session_type(**onnx_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_p2(), hr_runtime_options),
                model_path=f'{settings.models_path}/vision/high_resolution/imagenet1k/torchvision/resnet50_{hr_input_sizes_x[1]}.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15})
        ),
        # pycls: classification regnetx400mf_224x224 expected_metric: 72.7% top-1 accuracy
        'cl-6121':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(reverse_channels=True, resize=hr_input_sizes[0], crop=hr_input_sizes[0]),
            session=onnx_session_type(**onnx_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), hr_runtime_options),
                model_path=f'{settings.models_path}/vision/high_resolution/imagenet1k/fbr-pycls/regnetx-400mf_{hr_input_sizes_x[0]}.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':None})
        ),
        'cl-6122':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(reverse_channels=True, resize=hr_input_sizes[1], crop=hr_input_sizes[1]),
            session=onnx_session_type(**onnx_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), hr_runtime_options),
                model_path=f'{settings.models_path}/vision/high_resolution/imagenet1k/fbr-pycls/regnetx-400mf_{hr_input_sizes_x[1]}.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':None})
        ),
        # pycls: classification regnetx800mf_224x224 expected_metric: 75.2% top-1 accuracy
        'cl-6131':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(reverse_channels=True, resize=hr_input_sizes[0], crop=hr_input_sizes[0]),
            session=onnx_session_type(**onnx_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), hr_runtime_options),
                model_path=f'{settings.models_path}/vision/high_resolution/imagenet1k/fbr-pycls/regnetx-800mf_{hr_input_sizes_x[0]}.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':None})
        ),
        'cl-6132':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(reverse_channels=True, resize=hr_input_sizes[1], crop=hr_input_sizes[1]),
            session=onnx_session_type(**onnx_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), hr_runtime_options),
                model_path=f'{settings.models_path}/vision/high_resolution/imagenet1k/fbr-pycls/regnetx-800mf_{hr_input_sizes_x[1]}.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':None})
        ),
        # pycls: classification regnetx1.6gf_224x224 expected_metric: 77.0% top-1 accuracy
        'cl-6141':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(reverse_channels=True, resize=hr_input_sizes[0], crop=hr_input_sizes[0]),
            session=onnx_session_type(**onnx_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), hr_runtime_options),
                model_path=f'{settings.models_path}/vision/high_resolution/imagenet1k/fbr-pycls/regnetx-1.6gf_{hr_input_sizes_x[0]}.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':None})
        ),
        'cl-6142':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(reverse_channels=True, resize=hr_input_sizes[1], crop=hr_input_sizes[1]),
            session=onnx_session_type(**onnx_session_cfg,
                runtime_options=utils.dict_update(settings.runtime_options_onnx_np2(), hr_runtime_options),
                model_path=f'{settings.models_path}/vision/high_resolution/imagenet1k/fbr-pycls/regnetx-1.6gf_{hr_input_sizes_x[1]}.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':None})
        ),
    }
    return pipeline_configs
