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
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_IMAGENET]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_IMAGENET]['input_dataset'],
        'postprocess': postproc_transforms.get_transform_classification()
    }

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################jai-devkit models###############################
        # mlperf model: classification ResNet50v1.5 (resnet50_v1.onnx) expected_metric: 76.456% top-1 accuracy
        # convert_reducemean_to_matmul is supposed convert reducemean to matmul, so that tidl can handle it
        # but there seems to be an issue in tidl after this optimizer is applied.
        # disabling if for now - this will cause 2 subgraphs to be created and reducemean will run in onnxruntime
        'cl-6010':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, input_mean=(123.675, 116.28, 103.53), input_scale=(1.0, 1.0, 1.0),
                tidl_onnx_model_optimizer={'convert_reducemean_to_matmul': True, 'apply_default_optimizers': False}),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/mlperf/resnet50_v1_shape.onnx'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':76.456}, model_shortlist=30, compact_name='resNet50V1p5-mlperf-onnx', shortlisted=True)
        ),
        # jai-devkit: classification mobilenetv1_224x224 expected_metric: 71.82% top-1 accuracy
        'cl-6060':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv/mobilenet_v1_20190906.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':71.82}, model_shortlist=None, compact_name='mobileNetV1', shortlisted=False)
        ),
        # jai-devkit: classification mobilenetv2_224x224 expected_metric: 72.13% top-1 accuracy
        'cl-6070':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv/mobilenet_v2_20191224.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.13}, model_shortlist=None, compact_name='mobileNetV2', shortlisted=False)
        ),
        # jai-devkit: classification mobilenetv2_224x224 expected_metric: 72.13% top-1 accuracy, QAT: 71.73%
        'cl-6078':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_quant_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_qat_v1(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv/mobilenet_v2_qat-p2_20201213.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.13}, model_shortlist=None, compact_name='mobileNetV2-qat', shortlisted=False)
        ),
        # jai-devkit: classification mobilenetv2_1p4_224x224 expected_metric: 75.22% top-1 accuracy, QAT: 75.22%
        'cl-6158':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_quant_session_cfg(settings, work_dir=work_dir, input_optimization=False),
                runtime_options=settings.runtime_options_onnx_qat_v1(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv/mobilenet_v2_1p4_qat-p2_20210112.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':75.22}, model_shortlist=None, compact_name='mobileNetV2-1p4-qat', shortlisted=False)
        ),
        # jai-devkit: classification mobilenetv3_small_lite expected_metric: 62.688% top-1 accuracy
        'cl-6480':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv/mobilenet_v3_lite_small_20210429.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':62.688}, model_shortlist=30, compact_name='mobv3-lite-small', shortlisted=True)
        ),
        # jai-devkit: classification mobilenetv3_small_lite_qat expected_metric: 61.836% top-1 accuracy
        'cl-6488':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_quant_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_qat_v1(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv/mobilenet_v3_lite_small_qat-p2_20210429.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':61.836}, model_shortlist=40, compact_name='mobv3-lite-small-qat', shortlisted=True)
        ),
        # jai-devkit: classification mobilenetv3_large_lite expected_metric: 72.122% top-1 accuracy
        'cl-6490':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv/mobilenet_v3_lite_large_20210507.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.122}, model_shortlist=30, compact_name='mobv3-lite-large', shortlisted=True)
        ),
        #################torchvision models#########################
        # torchvision: classification shufflenetv2_224x224 expected_metric: 69.36% top-1 accuracy
        'cl-6080':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/shufflenet_v2_x1.0.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':69.36}, model_shortlist=90, compact_name='shuffleNetV2', shortlisted=False)
        ),
        # torchvision: classification mobilenetv2_224x224 expected_metric: 71.88% top-1 accuracy
        'cl-6090':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':71.88}, model_shortlist=20, compact_name='mobileNetV2-tv', shortlisted=True)
        ),
        # torchvision: classification mobilenetv2_224x224 expected_metric: 71.88% top-1 accuracy, QAT: 71.31%
        'cl-6098':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_quant_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_qat_v1(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_qat-p2.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':71.31}, model_shortlist=40, compact_name='mobileNetV2-tv-qat', shortlisted=True)
        ),
        # torchvision: classification resnet18_224x224 expected_metric: 69.76% top-1 accuracy
        'cl-6100':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/resnet18.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':69.76}, model_shortlist=30, compact_name='resNet18', shortlisted=True)
        ),
        # torchvision: classification resnet50_224x224 expected_metric: 76.15% top-1 accuracy
        'cl-6110':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/resnet50.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=30, compact_name='resNet50', shortlisted=True)
        ),
        #################pingolh-hardnet models#########################
        'cl-6470':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/pingolh-hardnet/hardnet39ds.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.1}, model_shortlist=None, compact_name='harDNet39ds', shortlisted=False)
        ),
        'cl-6460':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/pingolh-hardnet/hardnet68ds.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':74.3}, model_shortlist=None, compact_name='harDNet68ds', shortlisted=False)
        ),
        'cl-6440':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/pingolh-hardnet/hardnet68.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.5}, model_shortlist=None, compact_name='harDNet68', shortlisted=False)
        ),
        'cl-6450':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/pingolh-hardnet/hardnet85.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':78.0}, model_shortlist=None, compact_name='harDNet85', shortlisted=False)
        ),
        #################pycls regnetx models#########################
        # pycls: classification regnetx200mf_224x224 expected_metric: 68.9% top-1 accuracy
        'cl-6360':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(reverse_channels=True),
            session=onnx_session_type(**sessions.get_onnx_bgr_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/fbr-pycls/regnetx-200mf.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':68.9}, model_shortlist=1, compact_name='regNetx-200mf', shortlisted=True, recommended=True)
        ),
        #################torchvision models#########################
        # torchvision: classification regnetx400mf_224x224 expected_metric: 72.834% top-1 accuracy
        'cl-6160':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/regnet_x_400mf_tv.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.834}, model_shortlist=20, compact_name='regNetX-400mf-tv', shortlisted=True)
        ),
        # torchvision: classification regnetx800mf_224x224 expected_metric: 75.212% top-1 accuracy
        'cl-6170':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/regnet_x_800mf_tv.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':75.212}, model_shortlist=20, compact_name='regNetX-800mf-tv', shortlisted=True)
        ),
        # pycls: classification regnetx1.6gf_224x224 expected_metric: 77.040% top-1 accuracy
        'cl-6180':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/regnet_x_1_6gf_tv.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':77.040}, model_shortlist=40, compact_name='regNetX-1.6gf-tv', shortlisted=True)
        ),
        #################################################################
        #       TFLITE MODELS
        ##################tensorflow models##############################
        # mlperf/tf1 model: classification mobilenet_v1_224x224 expected_metric: 71.676 top-1 accuracy
        'cl-0000':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/mlperf/mobilenet_v1_1.0_224.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':71.676}, model_shortlist=10, compact_name='mobileNetV1-mlperf', shortlisted=True, recommended=True)
        ),
        # mlperf/tf-edge model: classification mobilenet_edgetpu_224 expected_metric: 75.6% top-1 accuracy
        'cl-0080':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/mlperf/mobilenet_edgetpu_224_1.0_float.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':75.6}, model_shortlist=40, compact_name='mobileNet-edgeTPU-mlperf', shortlisted=True)
        ),
        # mlperf model: classification ResNet50v1.5 (resnet50_v1.tflite) expected_metric: 76.456% top-1 accuracy
        'cl-0160':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_mean=(123.675, 116.28, 103.53), input_scale=(1.0, 1.0, 1.0)),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/mlperf/resnet50_v1.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':76.456}, model_shortlist=30, compact_name='resNet50V1p5-mlperf', shortlisted=True)
        ),
        #########################tensorflow1.0 models##################################
        # tensorflow/models: classification mobilenetv2_224x224 quant expected_metric: 70.0% top-1 accuracy
        'cl-0218':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite_quant(),
            session=tflite_session_type(**sessions.get_tflite_quant_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v1_1.0_224_quant.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':70.0}, model_shortlist=None, compact_name='mobileNetV1-qat', shortlisted=False)
        ),
        # tensorflow/models: classification mobilenetv2_224x224 expected_metric: 71.9% top-1 accuracy
        'cl-0010':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v2_1.0_224.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':71.9}, model_shortlist=30, compact_name='mobileNetV2', shortlisted=True)
        ),
        # tf hosted models: classification squeezenet_1 expected_metric: 57.5% top-1 accuracy
        'cl-0020':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_mean=(123.68, 116.78, 103.94), input_scale=(1/255, 1/255, 1/255)),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/squeezenet.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':57.5}, model_shortlist=None, compact_name='SqueezeNet', shortlisted=False)
        ),
        # tensorflow/models: classification mobilenetv2_224x224 expected_metric: 75.0% top-1 accuracy
        'cl-0200':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v2_float_1.4_224.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':75.0}, model_shortlist=90, compact_name='mobileNetV2-1p4', shortlisted=False)
        ),
        # tf hosted models: classification inception_v1_224_quant expected_metric: 69.63% top-1 accuracy
        'cl-0038':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite_quant(),
            session=tflite_session_type(**sessions.get_tflite_quant_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/inception_v1_224_quant.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':69.63}, model_shortlist=90, compact_name='InceptionNetV1', shortlisted=False)
        ),
        # tf hosted models: classification inception_v3 expected_metric: 78% top-1 accuracy
        'cl-0040':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(342, 299),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/inception_v3.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':78.0}, model_shortlist=90, compact_name='InceptionNetV3', shortlisted=False)
        ),
        # tf hosted models: classification mnasnet expected_metric: 74.08% top-1 accuracy
        'cl-0070':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mnasnet_1.0_224.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':74.08}, model_shortlist=None, compact_name='mnasNet', shortlisted=False)
        ),
        # tf1 models: classification resnet50_v1 expected_metric: 75.2% top-1 accuracy
        'cl-0050':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_mean=(123.675, 116.28, 103.53), input_scale=(1.0, 1.0, 1.0)),
                runtime_options=settings.runtime_options_tflite_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/resnet50_v1.tflite'),
            model_info=dict(metric_reference={'accuracy_top1%':75.2}, model_shortlist=None, compact_name='resNet50V1', shortlisted=False)
        ),
        # TODO: is this model's input correct? shouldn't it be 299 according to the slim page?
        # tf1 models: classification resnet50_v2 expected_metric: 75.6% top-1 accuracy
        'cl-0060':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/resnet50_v2.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':75.6}, model_shortlist=None, compact_name='resNet50V2', shortlisted=False)
        ),
        # tensorflow/models: classification mobilenet_v3-large-minimalistic_224_1.0_float expected_metric: 72.3% top-1 accuracy
        'cl-0260':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v3-large-minimalistic_224_1.0_float.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':72.3}, model_shortlist=None, compact_name='mobv3-large-minimalistic', shortlisted=False)
        ),
        # tensorflow/models: classification mobilenet_v3-small-minimalistic_224_1.0_float expected_metric: 61.9% top-1 accuracy
        'cl-0270':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v3-small-minimalistic_224_1.0_float.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':61.9}, model_shortlist=None, compact_name='mobv3-small-minimalistic', shortlisted=False)
        ),
        #################efficinetnet & tpu models#########################
        # tensorflow/tpu: classification efficinetnet-lite0_224x224 expected_metric: 75.1% top-1 accuracy
        'cl-0130':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_p2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-lite0-fp32.tflite'),
            model_info=dict(metric_reference={'accuracy_top1%':75.1}, model_shortlist=30, compact_name='efficientNet-lite0', shortlisted=True)
        ),
        # tensorflow/tpu: classification efficinetnet-lite1_240x240 expected_metric: 76.7% top-1 accuracy
        'cl-0170':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(274, 240),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-lite1-fp32.tflite'),
            model_info=dict(metric_reference={'accuracy_top1%':76.7}, model_shortlist=90, compact_name='efficientNet-lite1', shortlisted=False)
        ),
        # tensorflow/tpu: classification efficinetnet-lite4_300x300 expected_metric: 81.5% top-1 accuracy
        'cl-0140':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(343, 300),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-lite4-fp32.tflite'),
            model_info=dict(metric_reference={'accuracy_top1%':81.5}, model_shortlist=40, compact_name='efficientNet-lite4', shortlisted=True)
        ),
        # tensorflow/tpu: classification efficientnet-edgetpu-S expected_metric: 77.23% top-1 accuracy
        'cl-0090':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-edgetpu-S_float.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':77.23}, model_shortlist=40, compact_name='efficientNet-edgeTPU-s', shortlisted=True)
        ),
        # tensorflow/tpu: classification efficientnet-edgetpu-M expected_metric: 78.69% top-1 accuracy
        'cl-0100':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(274, 240),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-edgetpu-M_float.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':78.69}, model_shortlist=90, compact_name='efficientNet-edgeTPU-m', shortlisted=False)
        ),
        # tensorflow/tpu: classification efficientnet-edgetpu-L expected_metric: 80.62% top-1 accuracy
        'cl-0190':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(343, 300),
            session=tflite_session_type(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-edgetpu-L_float.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':80.62}, model_shortlist=None, compact_name='efficientNet-edgeTPU-l', shortlisted=False)
        ),
        ###################################################################
        # complied for TVM - this model is repeated here and hard-coded to use tvmdlr session to generate an example tvmdlr artifact
        # torchvision: classification mobilenetv2_224x224 expected_metric: 71.88% top-1 accuracy
        'cl-3090':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=sessions.TVMDLRSession(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':71.88}, model_shortlist=10, compact_name='mobileNetV2-tv', shortlisted=True, recommended=True)
        ),
        # torchvision: classification mobilenetv2_224x224 expected_metric: 71.88% top-1 accuracy, QAT: 71.31%
        'cl-3098':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=sessions.TVMDLRSession(**sessions.get_onnx_quant_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_qat_v1(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_qat-p2.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':71.31}, model_shortlist=None, compact_name='mobileNetV2-tv-qat', shortlisted=False)
        ),
        # torchvision: classification resnet50_224x224 expected_metric: 76.15% top-1 accuracy
        'cl-3110':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=sessions.TVMDLRSession(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/resnet50.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.15}, model_shortlist=None, compact_name='resNet50', shortlisted=False)
        ),
        # tensorflow/models: classification mobilenetv1_224x224 expected_metric: 71.0% top-1 accuracy (or is it 71.676% as this seems same as mlperf model)
        'cl-3520':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=sessions.TVMDLRSession(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_tflite_np2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v1_1.0_224.tflite'),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':71.0}, model_shortlist=None, compact_name='mobileNetV1', shortlisted=False)
        ),
        # tf1 models: classification resnet50_v1 expected_metric: 75.2% top-1 accuracy
        'cl-3530':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=sessions.TVMDLRSession(**sessions.get_tflite_session_cfg(settings, work_dir=work_dir, input_mean=(123.675, 116.28, 103.53), input_scale=(1.0, 1.0, 1.0)),
                runtime_options=settings.runtime_options_tflite_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/resnet50_v1.tflite'),
            model_info=dict(metric_reference={'accuracy_top1%':75.2}, model_shortlist=None, compact_name='resNet50-v1', shortlisted=False)
        ),
    }
    return pipeline_configs
