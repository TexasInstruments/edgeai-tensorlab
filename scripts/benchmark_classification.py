import os
from jacinto_ai_benchmark import *

# the cwd must be the root of the respository
if os.path.split(os.getcwd())[-1] == 'scripts':
    os.chdir('../')
#
# make sure current directory is visible for python import
if not os.environ['PYTHONPATH'].startswith(':'):
    os.environ['PYTHONPATH'] = ':' + os.environ['PYTHONPATH']
#

import config_settings as config
work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(__file__))[0], f'{config.tidl_tensor_bits}bits')
print(f'work_dir = {work_dir}')


################################################################################################
# configs for each model pipeline
common_cfg = {
    'type':'accuracy',
    'run_import':config.run_import,
    'run_inference':config.run_inference,
    'calibration_dataset':datasets.ImageNetCls(**config.imagenet_train_cfg),
    'input_dataset':datasets.ImageNetCls(**config.imagenet_val_cfg),
    'postprocess':config.get_postproc_classification()
}

pipeline_configs = [
    #################################################################
    #       ONNX MODELS
    #################jai-devkit models##############################
    # jai-devkit: classification mobilenetv1_224x224 expected_metric: 71.82% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_vgg(),
        'session':sessions.TVMDLRSession(**config.session_tvm_dlr_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/pytorch-jai-devkit/mobilenet_v1_20190906-171544_opset9.onnx')
    }),
    # jai-devkit: classification mobilenetv2_224x224 expected_metric: 72.13% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_vgg(),
        'session':sessions.TVMDLRSession(**config.session_tvm_dlr_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/pytorch-jai-devkit/mobilenet_v2_20191224-153212_opset9.onnx')
    }),
    # jai-devkit: classification mobilenetv2_224x224 expected_metric: 72.13% top-1 accuracy, QAT: 71.73%
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_vgg(),
        'session':sessions.TVMDLRSession(**config.session_tvm_dlr_cfg_qat, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/pytorch-jai-devkit/mobilenet_v2_qat-jai_20201213-165307_opset9.onnx')
    }),
    #################pycls regnetx models#########################
    # pycls: classification regnetx200mf_224x224 expected_metric: 68.9% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_vgg(reverse_channels=True),
        'session':sessions.TVMDLRSession(**config.session_tvm_dlr_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/pycls/RegNetX-200MF_dds_8gpu_opset9.onnx')
    }),
    # pycls: classification regnetx400mf_224x224 expected_metric: 72.7% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_vgg(reverse_channels=True),
        'session':sessions.TVMDLRSession(**config.session_tvm_dlr_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/pycls/RegNetX-400MF_dds_8gpu_opset9.onnx')
    }),
    # pycls: classification regnetx800mf_224x224 expected_metric: 75.2% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_vgg(reverse_channels=True),
        'session':sessions.TVMDLRSession(**config.session_tvm_dlr_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/pycls/RegNetX-800MF_dds_8gpu_opset9.onnx')
    }),
    # pycls: classification regnetx1.6gf_224x224 expected_metric: 77.0% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_vgg(reverse_channels=True),
        'session':sessions.TVMDLRSession(**config.session_tvm_dlr_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/pycls/RegNetX-1.6GF_dds_8gpu_opset9.onnx')
    }),
    #################torchvision models#########################
    # torchvision: classification shufflenetv2_224x224 expected_metric: 69.36% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_vgg(),
        'session':sessions.TVMDLRSession(**config.session_tvm_dlr_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/torchvision/shufflenet_v2_x1.0_opset9.onnx')
    }),
    # torchvision: classification mobilenetv2_224x224 expected_metric: 71.88% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_vgg(),
        'session':sessions.TVMDLRSession(**config.session_tvm_dlr_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_opset9.onnx')
    }),
    # torchvision: classification mobilenetv2_224x224 expected_metric: 71.88% top-1 accuracy, QAT: 71.31%
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_vgg(),
        'session':sessions.TVMDLRSession(**config.session_tvm_dlr_cfg_qat, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_qat-jai_opset9.onnx')
    }),
    # torchvision: classification resnet18_224x224 expected_metric: 69.76% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_vgg(),
        'session':sessions.TVMDLRSession(**config.session_tvm_dlr_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/torchvision/resnet18_opset9.onnx')
    }),
    # torchvision: classification resnet50_224x224 expected_metric: 76.15% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_vgg(),
        'session':sessions.TVMDLRSession(**config.session_tvm_dlr_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/torchvision/resnet50_opset9.onnx')
    }),
    # torchvision: classification vgg16_224x224 expected_metric: 71.59% top-1 accuracy - too slow inference
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_vgg(),
        'session':sessions.TVMDLRSession(**config.session_tvm_dlr_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/torchvision/vgg16_opset9.onnx')
    }),
    # github onnx model: classification resnet18_v2 expected_metric: 69.70% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_vgg(),
        'session':sessions.TVMDLRSession(**config.session_tvm_dlr_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/github-onnx-models/resnet18-v2-7.onnx'),
    }),
    #################################################################
    #       TFLITE MODELS
    ##################tensorflow models##############################
    # tensorflow/models: classification mobilenetv1_224x224 expected_metric: 70.9% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v1_1.0_224.tflite'),
        'metric':dict(label_offset_pred=-1)
    }),
    # tensorflow/models: classification mobilenetv1_224x224 quant expected_metric: 70.0% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v1_1.0_224_quant.tflite'),
        'metric':dict(label_offset_pred=-1)
    }),
    # tensorflow/models: classification mobilenetv2_224x224 expected_metric: 71.9% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v2_1.0_224.tflite'),
        'metric':dict(label_offset_pred=-1)
    }),
    # tensorflow/models: classification mobilenetv2_224x224 quant expected_metric: 70.8% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v2_1.0_224_quant.tflite'),
        'metric':dict(label_offset_pred=-1)
    }),
    # tensorflow/models: classification mobilenetv2_224x224 expected_metric: 75.0% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v2_float_1.4_224.tflite'),
        'metric':dict(label_offset_pred=-1)
    }),
    #################gen-efficinetnet models#########################
    # tensorflow/tpu: classification efficinetnet-lite0_224x224 expected_metric: 75.1% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-lite0-fp32.tflite')
    }),
    # tensorflow/tpu: classification efficinetnet-lite1_240x240 expected_metric: 76.7% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(274, 240),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-lite1-fp32.tflite')
    }),
    # tensorflow/tpu: classification efficinetnet-lite2_260x260 expected_metric: 77.6% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(297, 260),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-lite2-fp32.tflite')
    }),
    # tensorflow/tpu: classification efficinetnet-lite4_300x300 expected_metric: 81.5% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(343, 300),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-lite4-fp32.tflite')
    }),
    # tensorflow/tpu: classification efficientnet-edgetpu-S expected_metric: 77.23% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-edgetpu-S_float.tflite'),
        'metric':dict(label_offset_pred=-1)
    }),
    # tensorflow/tpu: classification efficientnet-edgetpu-M expected_metric: 78.69% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(274, 240),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-edgetpu-M_float.tflite'),
        'metric':dict(label_offset_pred=-1)
    }),
    # tensorflow/tpu: classification efficientnet-edgetpu-L expected_metric: 80.62% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(343, 300),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf-tpu/efficientnet-edgetpu-L_float.tflite'),
        'metric':dict(label_offset_pred=-1)
    }),
    # tf hosted models: classification squeezenet_1 expected_metric: 49.0% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf1-models/squeezenet.tflite'),
        'metric':dict(label_offset_pred=-1)
    }),
    # tf hosted models: classification densenet expected_metric: 74.98% top-1 accuracy (from publication)
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf1-models/densenet.tflite'),
        'metric':dict(label_offset_pred=-1)
    }),
    # tf hosted models: classification inception_v1_224_quant expected_metric: 69.63% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf1-models/inception_v1_224_quant.tflite'),
        'metric':dict(label_offset_pred=-1)
    }),
    # tf hosted models: classification inception_v3 expected_metric: 78% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(342, 299),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf1-models/inception_v3.tflite'),
        'metric':dict(label_offset_pred=-1)
    }),
    # tf hosted models: classification mnasnet expected_metric: 74.08% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf1-models/mnasnet_1.0_224.tflite'),
        'metric':dict(label_offset_pred=-1)
    }),
    # tf hosted models: classification nasnet mobile expected_metric: 73.9% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf1-models/nasnet_mobile.tflite'),
        'metric':dict(label_offset_pred=-1)
    }),
    # mlperf model: classification resnet50_v1 (BGR input, caffe preproc) expected_metric: 74.9% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(mean=(123.675, 116.28, 103.53), scale=(1.0, 1.0, 1.0), reverse_channels=True),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf2-models/resnet50.tflite')
    }),
    # mlperf model: classification resnet50_v2 expected_metric: 76.0% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf2-models/resnet50v2.tflite'),
        'metric':dict(label_offset_pred=-1)
    }),
    # tf1_models: classification xception expected_metric: 79.0% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(342, 299),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/classification/imagenet1k/tf2-models/xception.tflite')
    }),
    # mlperf model: classification resnet50_v1.5 expected_metric: 76.456% top-1 accuracy
    utils.dict_update(common_cfg, {
        'preprocess':config.get_preproc_inception(mean=(123.675, 116.28, 103.53), scale=(1.0, 1.0, 1.0)),
        'session':sessions.TFLiteRTSession(**config.session_tflite_rt_cfg, work_dir=work_dir,
            model_path=f'{config.modelzoo_path}/vision/mlperf/edge/resnet50-v1.5.tflite'),
        'metric':dict(label_offset_pred=-1)
    }),
]


################################################################################################
# execute each model
if __name__ == '__main__':
    if config.run_import or config.run_inference:
        pipelines.run(pipeline_configs, devices=config.cuda_devices)
    #
    results = pipelines.collect_results(work_dir)
    print(*results, sep='\n')


