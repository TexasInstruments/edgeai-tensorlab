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
        # edgeai-torchvison: classification mobilenetv2_lite 224x224 expected_metric: 72.8% top-1 accuracy
        'cl-6500':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/mobilenet_v2_lite_wt-v2_20231101_model.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.8}, model_shortlist=100)
        ),
        # edgeai-torchvison: classification mobilenetv3_large_lite_224x224 expected_metric: 71.7% top-1 accuracy
        'cl-6510':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/mobilenet_v3_large_lite_wt-v2_20231011_model.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':71.7}, model_shortlist=100)
        ),
        # edgeai-torchvision: classification resnet18_224x224 expected_metric: 69.76% top-1 accuracy
        'cl-6520':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/resnet18_lite_wt-v1_20230920.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':69.76}, model_shortlist=None)
        ),
        # edgeai-torchvision: classification resnet50_lite_224x224 expected_metric: 80.86% top-1 accuracy
        'cl-6530':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/resnet50_lite_wt-v2_20230919.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':80.86}, model_shortlist=100)
        ),
        # edgeai-torchvision: classification resnet101_224x224 expected_metric: 81.88% top-1 accuracy
        'cl-6540':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/resnet101_lite_wt-v2_20230919.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':81.88}, model_shortlist=100)
        ),
        # edgeai-torchvision: classification resnext50_224x224 expected_metric: 81.2% top-1 accuracy
        'cl-6550':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/resnext50_32x4d_lite_wt-v2_20230920.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':81.2}, model_shortlist=100)
        ),
        # edgeai-torchvision: classification resnext101_224x224 expected_metric: 82.83% top-1 accuracy
        'cl-6560':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/resnext101_32x8d_lite_wt-v2_20230920.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':82.83}, model_shortlist=100)
        ),
        # edgeai-torchvision: classification regnet_x_1_6gf_lite_224x224 expected_metric: 79.67% top-1 accuracy
        'cl-6570':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/regnet_x_1_6gf_lite_wt-v2_20230920.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':79.67}, model_shortlist=100)
        ),
        # edgeai-torchvision: classification regnet_x_400mf_lite101_224x224 expected_metric: 74.86% top-1 accuracy
        'cl-6580':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/regnet_x_400mf_lite_wt-v2_20230920.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':74.86}, model_shortlist=100)
        ),
        # edgeai-torchvision: classification regnet_x_800mf_lite_224x224 expected_metric: 75.21% top-1 accuracy
        'cl-6590':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/regnet_x_800mf_lite_wt-v2_20230920.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':75.21}, model_shortlist=100)
        ),
        ################################## deit/swin ##################
        'cl-6600':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'/data/hdd/users/a0508577/work/other/varun_model_classification/deit_tiny_1.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':75.21}, model_shortlist=None)
        ),
        'cl-6610':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'/data/hdd/users/a0508577/work/other/varun_model_classification/deit_small_patch16_224_sim.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':75.21}, model_shortlist=None)
        ),
        'cl-6620':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'/data/hdd/users/a0508577/work/other/varun_model_classification/deit3_small_patch16_224_sim.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':75.21}, model_shortlist=None)
        ),
        'cl-6630':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'/data/hdd/users/a0508577/work/other/varun_model_classification/swin_tiny_1.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':75.21}, model_shortlist=None)
        ),
        'cl-6640':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'/data/hdd/users/a0508577/work/other/varun_model_classification/swin_base_patch4_window12_384_1.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':75.21}, model_shortlist=None)
        ),
        'cl-6650':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'/data/hdd/users/a0508577/work/other/varun_model_classification/swinv2_tiny_window8_256_sim.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':75.21}, model_shortlist=None)
        ),
        'cl-6660':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'/data/hdd/users/a0508577/work/other/varun_model_classification/swinv2_base_window12to24_192to384_22kft1k.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':75.21}, model_shortlist=None)
        ),
        ################################## QAT models using Pytorch native QAT ##########################
        # edgeai-torchvison: classification mobilenetv2_224x224 pytorch-qat-perchannel expected_metric: 72.476% top-1 accuracy
        'cl-6507':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, with_onnxsim=True),
                runtime_options=settings.runtime_options_onnx_qat_v2(quantization_scale_type=constants.QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN, **quant_params_proto_path_disable_option),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/mobilenet_v2_lite_wt-v2_qat-v2-wc8-at8_20231120_model.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.476}, model_shortlist=20)
        ),
        # edgeai-torchvison: classification mobilenetv2_224x224 pytorch-qat-symm-power2 expected_metric: 72.184% top-1 accuracy
        'cl-6508':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, with_onnxsim=True),
                runtime_options=settings.runtime_options_onnx_qat_v2(quantization_scale_type=constants.QUANTScaleType.QUANT_SCALE_TYPE_P2, **quant_params_proto_path_disable_option),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/mobilenet_v2_lite_wt-v2_qat-v2-wt8sp2-at8sp2_20231124_model.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.184}, model_shortlist=20)
        ),
    }
    return pipeline_configs
