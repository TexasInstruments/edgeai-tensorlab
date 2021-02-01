import os
from jacinto_ai_benchmark import *

# the cwd must be the root of the respository
if os.path.split(os.getcwd())[-1] == 'scripts':
    os.chdir('../')
#

work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(__file__))[0])

################################################################################################
# setup parameters for each model

preprocess_tflite_rt = (preprocess.ImageRead(), preprocess.ImageResize(256),
                preprocess.ImageCenterCrop(224), preprocess.ImageToNumpyTensor4D(data_layout=constants.NHWC),
                preprocess.ImageNormMeanScale(mean=defaults.input_mean_127p5, scale=defaults.input_scale_127p5,
                                              data_layout=constants.NHWC))

preprocess_tvm_dlr = (preprocess.ImageRead(), preprocess.ImageResize(256),
                preprocess.ImageCenterCrop(224), preprocess.ImageToNumpyTensor4D(),
                preprocess.ImageNormMeanScale(mean=defaults.input_mean_imagenet, scale=defaults.input_scale_imagenet))

postprocess_classification = (postprocess.IndexArray(), postprocess.ArgMax())

pipeline_cfg = dict(
    type='accuracy',
    calibration_dataset=datasets.ImageNetClassification(**defaults.imagenet_train_cfg),
    input_dataset=datasets.ImageNetClassification(**defaults.imagenet_val_cfg),
    postprocess=postprocess_classification
)

session_cfg = dict(work_dir=work_dir,
                   tidl_tensor_bits=defaults.tidl_tensor_bits)

################################################################################################
# configs for each model pipeline

pipeline_configs = [
    utils.dict_update(pipeline_cfg,  # mlperf_mobilenet_v1_1.0_224 71.646% top-1 accuracy
        preprocess=preprocess_tflite_rt,
        session=sessions.TFLiteRTSession(**session_cfg,
             model_path=f'{defaults.modelzoo_path}/mlperf/edge/mlperf_mobilenet_v1_1.0_224.tflite',
             tidl_calibration_options=defaults.tidl_calibration_options_tflite,
             input_shape={'input': (1, 224, 224, 3)}),
        metric=dict(label_offset_pred=-1)),
    utils.dict_update(pipeline_cfg,  # mobilenet_v2_2019-12-24_15-32-12 72.13% top-1 accuracy
        preprocess=preprocess_tvm_dlr,
        session=sessions.TVMDLRSession(**session_cfg,
            model_path=f'./dependencies/examples/models/mobilenet_v2_2019-12-24_15-32-12_opset9.onnx',
            tidl_calibration_options=defaults.tidl_calibration_options_tvm,
            input_shape = {'input.1':(1, 3, 224, 224)}))
]

################################################################################################
# execute each model
if __name__ == '__main__':
    pipelines.run(pipeline_configs, devices=defaults.cuda_devices)
    results = pipelines.collect_results(work_dir)
    print(*results, sep='\n')


