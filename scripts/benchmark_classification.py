import os
from pytidl_benchmark.analysis import accuracy_benchmarks, collect_results
from pytidl_benchmark.utils import dict_update
from pytidl_benchmark.model.pytidl_model import PyTIDLModel
# from pytidl_benchmark.model.onnxrt_model import ONNXRTModel
# from pytidl_benchmark.model.tflite_model import TFLiteModel
# from pytidl_benchmark.model.tvm_model import TVMModel

# the cwd must be the root of the respository
if os.path.split(os.getcwd())[-1] == 'scripts':
    os.chdir('../')
#

numFrames = 50000 #1000 #10000 #50000
modelzoo_path = '../jacinto-ai-modelzoo/models'
datasets_path = f'./dependencies/datasets'
cuda_devices = [0,1,2,3] #None

common_cfg = dict(
    taskType='classification', workDirs='./work_dirs',
    numParamBits=8, numFramesCalibration=100, calibrationOption=7, #0, #7,
    inDataCalibration=dict(path=f'{datasets_path}/imagenet/train', split=f'{datasets_path}/imagenet/train.txt', shuffle=True),
    inData=dict(path=f'{datasets_path}/imagenet/val', split=f'{datasets_path}/imagenet/val.txt', shuffle=True),
    inDataFormat=1, inDataNorm=1, inMean=(123.675, 116.28, 103.53), inScale=(0.017125, 0.017507, 0.017429),
    resizeWidth=256, resizeHeight=256, inWidth=224, inHeight=224, inResizeType=1,  # KeepAR
    numFrames=min(numFrames, 50000),
)

common_cfg_260x260 = dict_update(
    common_cfg, resizeWidth=297, resizeHeight=297, inWidth=260, inHeight=260,
)

common_cfg_300x300 = dict_update(
    common_cfg, resizeWidth=343, resizeHeight=343, inWidth=300, inHeight=300,
)

common_cfg_bgr = dict_update(
    common_cfg, inDataFormat=0, inDataNorm=1, inMean=(103.53, 116.28, 123.675), inScale=(0.017429, 0.017507, 0.017125),
)

model_configs = [
    #################jai-devkit models#########################
    dict_update(common_cfg,  # jai-devkit: classification mobilenetv1_224x224 expected_metric: 71.82% top-1 accuracy
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/pytorch-jacinto-ai-devkit/mobilenet_v1_2019-09-06_17-15-44_opset9.onnx'),
    dict_update(common_cfg,  # jai-devkit: classification mobilenetv2_224x224 expected_metric: 72.13% top-1 accuracy
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/pytorch-jacinto-ai-devkit/mobilenet_v2_2019-12-24_15-32-12_opset9.onnx'),
    dict_update(common_cfg,  # jai-devkit: classification mobilenetv2_224x224 expected_metric: 72.13% top-1 accuracy, QAT: 71.73%
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/pytorch-jacinto-ai-devkit/mobilenet_v2_qat-jai_2020-12-13_16-53-07_opset9.onnx',
        calibrationOption=0, inResampleBackend='pillow'),
    #################pycls regnetx models#########################
    dict_update(common_cfg_bgr,  # pycls: classification regnetx200mf_224x224 expected_metric: 68.9% top-1 accuracy
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/pycls/RegNetX-200MF_dds_8gpu_opset9.onnx'),
    dict_update(common_cfg_bgr,  # pycls: classification regnetx400mf_224x224 expected_metric: 72.7% top-1 accuracy
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/pycls/RegNetX-400MF_dds_8gpu_opset9.onnx'),
    dict_update(common_cfg_bgr,  # pycls: classification regnetx800mf_224x224 expected_metric: 75.2% top-1 accuracy
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/pycls/RegNetX-800MF_dds_8gpu_opset9.onnx'),
    dict_update(common_cfg_bgr,  # pycls: classification regnetx1.6gf_224x224 expected_metric: 77.0% top-1 accuracy
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/pycls/RegNetX-1.6GF_dds_8gpu_opset9.onnx'),
    #################torchvision models#########################
    dict_update(common_cfg,  # torchvision: classification shufflenetv2_224x224 expected_metric: 69.36% top-1 accuracy
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/torchvision/shufflenetv2_x1p0_opset9.onnx'),
    dict_update(common_cfg,  # torchvision: classification mobilenetv2_224x224 expected_metric: 71.88% top-1 accuracy
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/torchvision/mobilenetv2_tv_x1_opset9.onnx'),
    dict_update(common_cfg,  # torchvision: classification mobilenetv2_224x224 expected_metric: 71.88% top-1 accuracy, QAT: 71.31%
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/torchvision/mobilenet_v2_tv_x1_qat-jai_opset9.onnx',
        calibrationOption=0, inResampleBackend='pillow'),
    dict_update(common_cfg,  # torchvision: classification resnet18_224x224 expected_metric: 69.76% top-1 accuracy
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/torchvision/resnet18_opset9.onnx'),
    dict_update(common_cfg,  # torchvision: classification resnet50_224x224 expected_metric: 76.15% top-1 accuracy
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/torchvision/resnet50_opset9.onnx'),
    dict_update(common_cfg, # torchvision: classification vgg16_224x224 expected_metric: 71.59% top-1 accuracy - too slow inference
       inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/torchvision/vgg16_opset9.onnx'),
    ##################gen-efficinetnet models#########################
    dict_update(common_cfg, # tensorflow/tpu: classification efficinetnet-lite0_224x224 expected_metric: 75.1% top-1 accuracy
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/tf_tpu/efficientnet-lite0.link',
        outDataNamesList='efficientnet-lite0/model/head/dense/BiasAdd'),
    dict_update(common_cfg_260x260, # tensorflow/tpu: classification efficinetnet-lite2_260x260 expected_metric: 77.6% top-1 accuracy
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/tf_tpu/efficientnet-lite2.link',
        outDataNamesList='efficientnet-lite2/model/head/dense/BiasAdd'),
    dict_update(common_cfg_300x300, # tensorflow/tpu: classification efficinetnet-lite4_300x300 expected_metric: 81.5% top-1 accuracy
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/tf_tpu/efficientnet-lite4.link',
        outDataNamesList='efficientnet-lite4/model/head/dense/BiasAdd'),
    ##################tensorflow models#########################
    dict_update(common_cfg,  # tensorflow/models: classification mobilenetv2_224x224 expected_metric: 70.9% top-1 accuracy
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/tf1_models/mobilenet_v1_float_1.0_224.link',
        outDataNamesList='MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd', labelOffset=1),
    dict_update(common_cfg,  # tensorflow/models: classification mobilenetv2_224x224 expected_metric: 71.9% top-1 accuracy
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/tf1_models/mobilenet_v2_float_1.0_224.link',
        outDataNamesList='MobilenetV2/Logits/Conv2d_1c_1x1/BiasAdd', labelOffset=1),
    dict_update(common_cfg, # tensorflow/models: classification mobilenetv2_224x224 expected_metric: 75.0% top-1 accuracy
        inputNetFile=f'{modelzoo_path}/edge/classification/imagenet1k/tf1_models/mobilenet_v2_float_1.4_224.link',
        outDataNamesList='MobilenetV2/Logits/Conv2d_1c_1x1/BiasAdd', labelOffset=1),
]


if __name__ == '__main__':
    workDirs = common_cfg.pop('workDirs', None)
    workDirs = os.path.abspath(workDirs) if workDirs is not None else workDirs
    expt_name = os.path.splitext(os.path.basename(__file__))[0]
    accuracy_benchmarks(model_type=PyTIDLModel, model_configs=model_configs, workDirs=f'{workDirs}/{expt_name}', devices=cuda_devices)
    results = collect_results(f'{workDirs}/{expt_name}')
    print(*results, sep='\n')


