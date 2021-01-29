import os
from pytidl_benchmark.analysis import accuracy_benchmarks, collect_results
from pytidl_benchmark.utils import dict_update

# the cwd must be the root of the respository
if os.path.split(os.getcwd())[-1] == 'scripts':
    os.chdir('../')
#

numFrames = 50000 #1000 #10000 #50000
modelzoo_path = './dependencies/modelzoo/jai-modelzoo'
datasets_path = f'./dependencies/datasets'
cuda_devices = None #[0,1,2,3] #None

common_cfg = dict(
    taskType='classification',
    workDirs='./work_dirs',
    numParamBits=8, #8, #16, #32,
    numFramesCalibration=100,
    calibrationOption=0, #0, #7,
    inDataCalibration=dict(path=f'{datasets_path}/imagenet/train', split=f'{datasets_path}/imagenet/train.txt', shuffle=True),
    inData=dict(path=f'{datasets_path}/imagenet/val', split=f'{datasets_path}/imagenet/val.txt', shuffle=True),
    inDataFormat=1,
    inDataNorm=1,
    inMean=(123.675, 116.28, 103.53),
    inScale=(0.017125, 0.017507, 0.017429),
    inResizeType=1,  # KeepAR
    resizeWidth=256,
    resizeHeight=256,
    inWidth=224,
    inHeight=224,
    numFrames=min(numFrames, 50000),
)

model_configs = [
    dict_update(common_cfg,  # caffe-jacinto: classification jacintonet11v2 expected_metric: 60.9% top-1 accuracy
        inputNetFile=f'{modelzoo_path}/edge/caffe/classification/imagenet1k/caffe-jacinto/jacintonet11v2_deploy.prototxt',
        inputParamsFile=f'{modelzoo_path}/edge/caffe/classification/imagenet1k/caffe-jacinto/jacintonet11v2_iter320k.caffemodel',
        inDataFormat=0, inMean=(128.0, 128.0, 128.0), inScale=(1.0, 1.0, 1.0)),
]


if __name__ == '__main__':
    from pytidl_benchmark.model.pytidl_model import PyTIDLModel
    model_type = PyTIDLModel
    workDirs = common_cfg.pop('workDirs', None)
    workDirs = os.path.abspath(workDirs) if workDirs is not None else workDirs
    expt_name = os.path.splitext(os.path.basename(__file__))[0]
    accuracy_benchmarks(model_configs, model_type, f'{workDirs}/{expt_name}', devices=cuda_devices)
    results = collect_results(f'{workDirs}/{expt_name}')
    print(*results, sep='\n')


