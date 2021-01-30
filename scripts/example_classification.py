import os
from jacinto_ai_benchmark import analysis, datasets, preprocess, sessions, postprocess, metrics, utils

# the cwd must be the root of the respository
if os.path.split(os.getcwd())[-1] == 'scripts':
    os.chdir('../')
#

################################################################################################
# set common configuration params
num_frames = 10 #50000
num_frames_calibration = 100
modelzoo_path = './dependencies/modelzoo/jai-modelzoo'
datasets_path = f'./dependencies/datasets'
cuda_devices = None #[0,1,2,3] #None
tidl_dir = './dependencies/c7x-mma-tidl'
tidl_tensor_bits = 32
tidl_calibration_iterations = 50
work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(__file__))[0])
imagenet_train_cfg = dict(path=f'{datasets_path}/imagenet/train', split=f'{datasets_path}/imagenet/train.txt',
                          shuffle=True,num_frames=num_frames_calibration)
imagenet_val_cfg = dict(path=f'{datasets_path}/imagenet/val', split=f'{datasets_path}/imagenet/val.txt',
                        shuffle=True,num_frames=num_frames)
imagenet_mean = (123.675, 116.28, 103.53)
imagenet_scale = (0.017125, 0.017507, 0.017429)

utils.setup_environment(tidl_dir=tidl_dir)

################################################################################################
# setup parameters for each model

common_cfg = dict(
    calibration_dataset=datasets.ImageNetClassification(**imagenet_train_cfg),
    input_dataset=datasets.ImageNetClassification(**imagenet_val_cfg),
    preprocess=(preprocess.ImageRead(), preprocess.ImageResize(256),
                preprocess.ImageCenterCrop(224), preprocess.ImageToNumpyTensor4D(),
                preprocess.ImageNormMeanScale(mean=imagenet_mean, scale=imagenet_scale)),
    postprocess=(postprocess.IndexArray(),postprocess.ArgMax())
)

pipeline_configs = [
    utils.dict_update(common_cfg,  # mobilenet_v2_2019-12-24_15-32-12 72.13% top-1 accuracy
        session=sessions.TVMDLRSession(work_dir=work_dir,
            tidl_tensor_bits=tidl_tensor_bits, tidl_calibration_options=dict(iterations=tidl_calibration_iterations),
            model_path='./dependencies/examples/models/mobilenet_v2_2019-12-24_15-32-12_opset9.onnx',
            input_shape = {'input.1':(1, 3, 224, 224)}
        )
    )
]

################################################################################################
# execute each model
if __name__ == '__main__':
    analysis.run_pipelines(pipeline_configs, devices=cuda_devices)
    results = analysis.collect_results(work_dir)
    print(*results, sep='\n')


