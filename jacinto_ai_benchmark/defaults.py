from . import preprocess, postprocess, constants


###############################################################
# some utility functions to easily create the pipeline

def get_preproc_tvm_dlr(resize=256, crop=224, data_layout=constants.NCHW, reverse_channels=False, backend='pil',
                        mean=(123.675, 116.28, 103.53), scale=(0.017125, 0.017507, 0.017429)):
    preprocess_tvm_dlr = [
        preprocess.ImageRead(backend=backend),
        preprocess.ImageResize(resize),
        preprocess.ImageCenterCrop(crop),
        preprocess.ImageToNPTensor4D(data_layout=data_layout),
        preprocess.ImageNormMeanScale(mean=mean, scale=scale, data_layout=data_layout)]
    if reverse_channels:
        preprocess_tvm_dlr = preprocess_tvm_dlr + [preprocess.NPTensor4DChanReverse(data_layout=data_layout)]
    #
    return preprocess_tvm_dlr


def get_preproc_tflite_rt(resize=256, crop=224, data_layout=constants.NHWC, reverse_channels=False, backend='pil',
                          mean=(127.0, 127.0, 127.0), scale=(1/128.0, 1/128.0, 1/128.0)):
    preprocess_tflite_rt = [
        preprocess.ImageRead(backend=backend),
        preprocess.ImageResize(resize),
        preprocess.ImageCenterCrop(crop),
        preprocess.ImageToNPTensor4D(data_layout=data_layout),
        preprocess.ImageNormMeanScale(mean=mean, scale=scale, data_layout=data_layout)]
    if reverse_channels:
        preprocess_tflite_rt = preprocess_tflite_rt + [preprocess.NPTensor4DChanReverse(data_layout=data_layout)]
    #
    return preprocess_tflite_rt


def get_postproc_classification():
    postprocess_classification = [postprocess.IndexArray(), postprocess.ArgMax()]
    return postprocess_classification

