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


# for transformer models we need to set graph_optimization_level = ORT_DISABLE_ALL for onnxruntime
from onnxruntime import GraphOptimizationLevel
ORT_DISABLE_ALL = GraphOptimizationLevel.ORT_DISABLE_ALL

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

    quant_params_proto_path_disable_option = {constants.ADVANCED_OPTIONS_QUANT_FILE_KEY: ''}

    pipeline_configs = {
        #################################################################
        #       ONNX MODELS
        #################classification###############################
        ################################## QAT models using Pytorch native QAT ##########################
        # edgeai-torchvison: classification mobilenetv2_224x224 pytorch-qat-perchannel expected_metric: 72.476% top-1 accuracy
        'cl-6508':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, with_onnxsim=True),
                runtime_options=settings.runtime_options_onnx_qat_v2(fast_calibration=False, **quant_params_proto_path_disable_option),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/mobilenet_v2_lite_wt-v2_qat-v2-wc8-at8_20231120_model.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.476}, model_shortlist=20, compact_name='mobileNetV2-lite-wtv2-qatv2-perc-224', shortlisted=False)
        ),
        # edgeai-torchvison: classification mobilenetv2_224x224 pytorch-qat-symm-power2 expected_metric: 72.184% top-1 accuracy
        'cl-6507':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, with_onnxsim=True),
                runtime_options=settings.runtime_options_onnx_qat_v2_p2(fast_calibration=False, **quant_params_proto_path_disable_option),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/mobilenet_v2_lite_wt-v2_qat-v2-wt8sp2-at8sp2_20231124_model.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.184}, model_shortlist=None, compact_name='mobileNetV2-lite-wtv2-qatv2-pert-sp2-224', shortlisted=False)
        ),
        ################################## float models ##########################                
        # edgeai-torchvison: classification mobilenetv2_lite 224x224 expected_metric: 72.8% top-1 accuracy
        'cl-6500':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/mobilenet_v2_lite_wt-v2_20231101_model.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.8}, model_shortlist=100, compact_name='mobileNetV2-lite-wtv2-224', shortlisted=False)
        ),
        # edgeai-torchvison: classification mobilenetv3_large_lite_224x224 expected_metric: 71.7% top-1 accuracy
        'cl-6510':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/mobilenet_v3_large_lite_wt-v2_20231011_model.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':71.7}, model_shortlist=100, compact_name='mobileNetV3-large-lite-wtv2-224', shortlisted=False)
        ),
        # edgeai-torchvision: classification resnet18_224x224 expected_metric: 69.76% top-1 accuracy
        'cl-6520':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/resnet18_lite_wt-v1_20230920.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':69.76}, model_shortlist=None, compact_name='resNet18-wtv1-224', shortlisted=False)
        ),
        # edgeai-torchvision: classification resnet50_lite_224x224 expected_metric: 80.86% top-1 accuracy
        'cl-6530':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/resnet50_lite_wt-v2_20230919.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':80.86}, model_shortlist=100, compact_name='resNet50-wtv2-224', shortlisted=False)
        ),
        # edgeai-torchvision: classification resnet101_224x224 expected_metric: 81.88% top-1 accuracy
        'cl-6540':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/resnet101_lite_wt-v2_20230919.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':81.88}, model_shortlist=100, compact_name='resNet101-wtv2-224', shortlisted=False)
        ),
        # edgeai-torchvision: classification resnext50_224x224 expected_metric: 81.2% top-1 accuracy
        'cl-6550':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/resnext50_32x4d_lite_wt-v2_20230920.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':81.2}, model_shortlist=100, compact_name='resNeXt50-32x4d-wtv2-224', shortlisted=False)
        ),
        # edgeai-torchvision: classification resnext101_224x224 expected_metric: 82.83% top-1 accuracy
        'cl-6560':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/resnext101_32x8d_lite_wt-v2_20230920.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':82.83}, model_shortlist=100, compact_name='resNeXt101-32x8d-wtv2-224', shortlisted=False)
        ),
        # edgeai-torchvision: classification regnet_x_1_6gf_lite_224x224 expected_metric: 79.67% top-1 accuracy
        'cl-6570':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/regnet_x_1_6gf_lite_wt-v2_20230920.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':79.67}, model_shortlist=100, compact_name='regNetX-1.6gf-wtv2-224', shortlisted=False)
        ),
        # edgeai-torchvision: classification regnet_x_400mf_lite101_224x224 expected_metric: 74.86% top-1 accuracy
        'cl-6580':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/regnet_x_400mf_lite_wt-v2_20230920.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':74.86}, model_shortlist=100, compact_name='regNetX-400mf-wtv2-224', shortlisted=False)
        ),
        # edgeai-torchvision: classification regnet_x_800mf_lite_224x224 expected_metric: 75.21% top-1 accuracy
        'cl-6590':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/regnet_x_800mf_lite_wt-v2_20230920.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':75.21}, model_shortlist=100, compact_name='regNetX-800mf-wtv2-224', shortlisted=False)
        ),      
        # edgeai-torchvison: classification mobilenetv2 224x224 expected_metric: 72.15% top-1 accuracy
        'cl-6600':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_np2(fast_calibration=False),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/edgeai-tv2/mobilenet_v2_wt-v2_20240904_model.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.15}, model_shortlist=100, compact_name='mobileNetV2-wtv2-224', shortlisted=False)
        ),        

        ################################## huggingface transformer models ##################
        'cl-6700':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer={'hf_attention_block_optimization':True}),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/vit_tiny_patch16_224_simp.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':45.23}, model_shortlist=80, compact_name='ViT-tiny-patch16-transformer-224', shortlisted=False)
        ),
        # inference takes too much time - limit th number of frames for inference
        'cl-6701':utils.dict_update(utils.dict_update(common_cfg, num_frames=(min(settings.num_frames,1000) if settings.num_frames else None)),
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer={'hf_attention_block_optimization':True}),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/vit_base_patch16_224_simp.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':75.40}, model_shortlist=80, compact_name='ViT-base-patch16-transformer-224', shortlisted=False)
        ),          
        'cl-6710':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer={'hf_attention_block_optimization':True}),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/deit_tiny_patch16_224_simp.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':72.13}, model_shortlist=80, compact_name='DeiT-tiny-patch16-transformer-224', shortlisted=False)
        ),
        'cl-6711':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer={'hf_attention_block_optimization':True}),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/deit_small_patch16_224_simp.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':79.9}, model_shortlist=80, compact_name='DeiT-small-patch16-transformer-224', shortlisted=False)
        ),
        'cl-6720':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer={'hf_attention_block_optimization':True}),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/levit_128_224_simp.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':78.59}, model_shortlist=80, compact_name='LeViT-128-transformer-224', shortlisted=False)
        ),  
        'cl-6721':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer={'hf_attention_block_optimization':True}),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/levit_256_224_simp.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':81.59}, model_shortlist=80, compact_name='LeViT-256-transformer-224', shortlisted=False)
        ),          
        'cl-6730':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer={'hf_attention_block_optimization':True, 'expand_slice_across_multiple_axis':True, 'apply_default_optimizers':False}),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/swin_tiny_patch4_window7_224_simp.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':80.43}, model_shortlist=80, compact_name='Swin-tiny-patch4-window7-transformer-224', shortlisted=False)
        ),  
        'cl-6731':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer={'hf_attention_block_optimization':True, 'expand_slice_across_multiple_axis':True, 'apply_default_optimizers':False}),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/swin_small_patch4_window7_224_simp.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':83.07}, model_shortlist=80, compact_name='Swin-small-patch4-window7-transformer-224', shortlisted=False)
        ),         
        'cl-6732':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer={'hf_attention_block_optimization':True, 'expand_slice_across_multiple_axis':True, 'apply_default_optimizers':False}),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/swin_base_patch4_window7_224_simp.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':84.81}, model_shortlist=80, compact_name='Swin-base-patch4-window7-transformer-224', shortlisted=False)
        ),   
        # torchvision: classification convnext_tiny expected_metric:  top-1 accuracy
        'cl-6740':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, with_onnxsim=True, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'../edgeai-modelzoo/models/vision/classification/imagenet1k/torchvision/convnext_tiny.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':82.52}, model_shortlist=110, compact_name='convNext-tiny-tv-224', shortlisted=False)
        ),
        # torchvision: classification convnext_small expected_metric:  top-1 accuracy
        'cl-6741':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, with_onnxsim=True, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'../edgeai-modelzoo/models/vision/classification/imagenet1k/torchvision/convnext_small.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':83.63}, model_shortlist=110, compact_name='convNext-small-tv-224', shortlisted=False)
        ),
        # torchvision: classification efficientnet_b0 expected_metric:  top-1 accuracy
        'cl-6750':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=256, crop=224),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, with_onnxsim=True, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(fast_calibration=False),
                model_path=f'../edgeai-modelzoo/models/vision/classification/imagenet1k/torchvision/efficientnet_b0.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':77.692}, model_shortlist=110, compact_name='efficientNet-b0-224', shortlisted=False)
        ),
        # torchvision: classification efficientnet_b1 expected_metric:  top-1 accuracy
        'cl-6751':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=255, crop=240),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, with_onnxsim=True, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(fast_calibration=False),
                model_path=f'../edgeai-modelzoo/models/vision/classification/imagenet1k/torchvision/efficientnet_b1.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':79.838}, model_shortlist=110, compact_name='efficientNet-b1-240', shortlisted=False)
        ),
        # torchvision: classification efficientnet_b2 expected_metric:  top-1 accuracy
        'cl-6752':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=288, crop=288),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, with_onnxsim=True, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(fast_calibration=False),
                model_path=f'../edgeai-modelzoo/models/vision/classification/imagenet1k/torchvision/efficientnet_b2.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':80.608}, model_shortlist=110, compact_name='efficientNet-b2-288', shortlisted=False)
        ),
        # torchvision: classification efficientnet_b3 expected_metric:  top-1 accuracy
        'cl-6753':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=320, crop=300),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, with_onnxsim=True, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(fast_calibration=False),
                model_path=f'../edgeai-modelzoo/models/vision/classification/imagenet1k/torchvision/efficientnet_b3.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':82.008}, model_shortlist=120, compact_name='efficientNet-b3-300', shortlisted=False)
        ),
        # torchvision: classification efficientnet_b4 expected_metric:  top-1 accuracy
        'cl-6754':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=384, crop=380),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, with_onnxsim=True, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(fast_calibration=False),
                model_path=f'../edgeai-modelzoo/models/vision/classification/imagenet1k/torchvision/efficientnet_b4.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':83.384}, model_shortlist=120, compact_name='efficientNet-b4-380', shortlisted=False)
        ),
        # torchvision: classification efficientnet_b5 expected_metric:  top-1 accuracy
        'cl-6755':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=456, crop=456),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, with_onnxsim=True, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(fast_calibration=False),
                model_path=f'../edgeai-modelzoo/models/vision/classification/imagenet1k/torchvision/efficientnet_b5.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':83.444}, model_shortlist=130, compact_name='efficientNet-b5-456', shortlisted=False)
        ),
        # torchvision: classification efficientnet_b6 expected_metric:  top-1 accuracy
        'cl-6756':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=528, crop=528),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, with_onnxsim=True, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(fast_calibration=False),
                model_path=f'../edgeai-modelzoo/models/vision/classification/imagenet1k/torchvision/efficientnet_b6.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':84.008}, model_shortlist=130, compact_name='efficientNet-b6-528', shortlisted=False)
        ),
        # torchvision: classification mobilenet_v3_small expected_metric:  top-1 accuracy
        'cl-6760':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, with_onnxsim=False, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(fast_calibration=False),
                model_path=f'../edgeai-modelzoo/models/vision/classification/imagenet1k/torchvision/mobilenet_v3_small_simp.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':67.668}, model_shortlist=110, compact_name='mobilenetV3-small-224', shortlisted=False)
        ),
        # torchvision: classification mobilenet_v3_large expected_metric:  top-1 accuracy
        'cl-6761':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, with_onnxsim=True, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(fast_calibration=False),
                model_path=f'../edgeai-modelzoo/models/vision/classification/imagenet1k/torchvision/mobilenet_v3_large.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':75.274}, model_shortlist=110, compact_name='mobilenetV3-large-224', shortlisted=False)
        ),  
        # hf-transformers: classification convnext-tiny-224 expected_metric:  top-1 accuracy
        'cl-6770':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, with_onnxsim=False, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'../edgeai-modelzoo/models/vision/classification/imagenet1k/hf-transformers/convnext-tiny-224_simp.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':82.1}, model_shortlist=110, compact_name='convNext-tiny-hf-224', shortlisted=False)
        ),
        #FAST-VIT
        'cl-6780':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(256,256),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/timm/fastvit_s12.apple_in1k_reparam_simp.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':78.59}, model_shortlist=110, compact_name='FastViT-s12-transformer-256', shortlisted=False)
        ),
        'cl-6781':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(256,256),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/timm/fastvit_sa12.apple_in1k_reparam_simp.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':78.59}, model_shortlist=110, compact_name='FastViT-s12-transformer-256', shortlisted=False)
        ),
        # MAX-VIT
        'cl-6790':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(224,224),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/maxvit_tiny_tf_224.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':83.49}, model_shortlist=130, compact_name='MaxVit-tiny-transformer-224', shortlisted=False)
        ),
        'cl-6791':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(224,224),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/maxvit_small_tf_224.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':84.46}, model_shortlist=130, compact_name='MaxVit-small-transformer-224', shortlisted=False)
        ),
        # Intern-Image
        'cl-6800':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(224,224),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/internimage_t_1k_224.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':83.24}, model_shortlist=130, compact_name='InternImage-tiny', shortlisted=False)
        ),
        'cl-6801':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(224,224),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/internimage_s_1k_224.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':83.96}, model_shortlist=130, compact_name='InternImage-small', shortlisted=False)
        ),
        # Visual Attention Network
        'cl-6810':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(224,224),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/van-tiny.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':70.44}, model_shortlist=120, compact_name='VAN-tiny', shortlisted=False)
        ),
        'cl-6811':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(224,224),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/van-small.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.08}, model_shortlist=120, compact_name='VAN-small', shortlisted=False)
        ),
        'cl-6812':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(224,224),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/van-base.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':80.84}, model_shortlist=120, compact_name='VAN-base', shortlisted=False)
        ),
        # CAFormer
        'cl-6820':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(224,224),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/caformer_s18_sail.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':83.58}, model_shortlist=120, compact_name='CAFormer-s18', shortlisted=False)
        ),
        'cl-6821':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(224,224),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/caformer_s36_sail.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':84.42}, model_shortlist=120, compact_name='CAFormer-s36', shortlisted=False)
        ),
        # EfficientViT
        'cl-6830':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(224,224),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/efficientvit_b0.r224_in1k.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':71.6}, model_shortlist=130, compact_name='EfficientVit_B0', shortlisted=False)
        ),
        'cl-6831':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(224,224),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/efficientvit_b1.r224_in1k.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':79.41}, model_shortlist=130, compact_name='EfficientVit_B1', shortlisted=False)
        ),
        'cl-6832':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(224,224),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/efficientvit_b2.r224_in1k.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':82.12}, model_shortlist=130, compact_name='EfficientVit_B2', shortlisted=False)
        ),
        'cl-6841':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(224,224),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/efficientvit_l1.r224_in1k.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':84.12}, model_shortlist=130, compact_name='EfficientVit_L1', shortlisted=False)
        ),
        'cl-6842':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(224,224),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/efficientvit_l2.r224_in1k.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':84.86}, model_shortlist=130, compact_name='EfficientVit_L2', shortlisted=False)
        ),
        'cl-6855':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(224,224),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir, input_optimization=False, tidl_onnx_model_optimizer=True),
                runtime_options=settings.runtime_options_onnx_np2(ext_options={'onnxruntime:graph_optimization_level': ORT_DISABLE_ALL}),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/hf-transformers/efficientvit_m5.r224_in1k.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':76.49}, model_shortlist=120, compact_name='EfficientVit_M5', shortlisted=False)
        ),
        # onnx-models: classification resnet50_v2 (pre-batchnorm) expected_metric: 75.81% top-1 accuracy
        'cl-6860':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(224,224),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/onnx-models/resnet50-v2-7.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':75.81}, model_shortlist=120, compact_name='ResNet50-V2-PreBN', shortlisted=False)
        ),
        'cl-6870':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=384, crop=384),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/efficientnet_v2_s.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':84.228}, model_shortlist=120, compact_name='EfficientNet-V2-S-384', shortlisted=False)
        ),
        'cl-6871':utils.dict_update(common_cfg,
            preprocess=preproc_transforms.get_transform_onnx(resize=480, crop=480),
            session=onnx_session_type(**sessions.get_onnx_session_cfg(settings, work_dir=work_dir),
                runtime_options=settings.runtime_options_onnx_p2(),
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/efficientnet_v2_m.onnx'),
            model_info=dict(metric_reference={'accuracy_top1%':85.112}, model_shortlist=120, compact_name='EfficientNet-V2-M-480', shortlisted=False)
        ),
    }
    return pipeline_configs
