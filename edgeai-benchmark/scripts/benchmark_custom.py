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

import os
import argparse
import cv2
from jai_benchmark import *


def get_imagecls_dataset_loaders(settings, download=False):
    # this example uses the datasets.ImageClassification data loader
    # this data loader assumes that the split argument provided is a text file containing a list of images
    # that are inside the folder provided in the path argument.
    # the split file also should contain the class id after a space.
    # so the split file format should be (for example)
    # image3.png 10
    # image2.jpg 1
    # image_cat.png 3
    # etc.
    # all the images need not be inside the path, but can be inside subdirectories,
    # but path combined with the lines in split file should give the image path.
    dataset_calib_cfg = dict(
        path=f'{settings.datasets_path}/imagenet/val',
        split=f'{settings.datasets_path}/imagenet/val.txt',
        num_classes=1000,
        shuffle=True,
        num_frames=min(settings.calibration_frames,50000),
        name='imagenet'
    )

    # dataset parameters for actual inference
    dataset_val_cfg = dict(
        path=f'{settings.datasets_path}/imagenet/val',
        split=f'{settings.datasets_path}/imagenet/val.txt',
        num_classes=1000,
        shuffle=True,
        num_frames=min(settings.num_frames,50000),
        name='imagenet'
    )

    # you are free to use any other data loaded provided in datasets folder or write your own instead of this
    calib_dataset = datasets.ImageClassification(**dataset_calib_cfg, download=download)
    val_dataset = datasets.ImageClassification(**dataset_val_cfg, download=download)
    return calib_dataset, val_dataset


def get_imageseg_dataset_loaders(settings, download=False):
    # this example uses the datasets.ImageSegmentation data loader
    # it assumes that the split is a text file listing input images and label images
    # that is each line should contain the image path and label image path
    # that are inside the folder provided in the path argument.
    dataset_calib_cfg = dict(
        path=f'{settings.datasets_path}/coco-seg21-converted/val2017',
        split=f'{settings.datasets_path}/coco-seg21-converted/val2017.txt',
        num_classes=21,
        shuffle=True,
        num_frames=min(settings.calibration_frames,5000),
        name='cocoseg21'
    )

    # dataset parameters for actual inference
    dataset_val_cfg = dict(
        path=f'{settings.datasets_path}/coco-seg21-converted/val2017',
        split=f'{settings.datasets_path}/coco-seg21-converted/val2017.txt',
        num_classes=21,
        shuffle=True,
        num_frames=min(settings.num_frames,5000),
        name='cocoseg21'
    )

    # you are free to use any other data loaded provided in datasets folder or write your own instead of this
    calib_dataset = datasets.ImageSegmentation(**dataset_calib_cfg, download=download)
    val_dataset = datasets.ImageSegmentation(**dataset_val_cfg, download=download)
    return calib_dataset, val_dataset


def get_imagedet_dataset_loaders(settings, download=False):
    # this example uses the datasets.COCODetection data loader
    dataset_calib_cfg = dict(
        path=f'{settings.datasets_path}/coco',
        split='val2017',
        shuffle=True,
        num_frames=min(settings.calibration_frames,5000))

    # dataset parameters for actual inference
    dataset_val_cfg = dict(
        path=f'{settings.datasets_path}/coco',
        split='val2017',
        shuffle=False, # can be set to True as well, if needed
        num_frames=min(settings.num_frames,5000))

    # you are free to use any other data loaded provided in datasets folder or write your own instead of this
    calib_dataset = datasets.COCODetection(**dataset_calib_cfg, download=download)
    val_dataset = datasets.COCODetection(**dataset_val_cfg, download=download)
    return calib_dataset, val_dataset


def create_configs(settings, work_dir):
    '''
    configs for each model pipeline
    - calibration_dataset: dataset to be used for import - should support __len__ and __getitem__.
    - input_dataset: dataset to be used for inference - should support __len__ and __getitem__
      Output of __getitem__ should be understood by the preprocess stage.
      For example, if the dataset returns image filenames, the first entry in the preprocess can be an image read class.
    - preprocess is just a list of transforms wrapped in utils.TransformsCompose.
      It depends on what the dataset class outputs and what the model expects.
      We have some default transforms defined in settings.
    - postprocess is also a list of transforms wrapped in utils.TransformsCompose
      It depends on what the model outputs and what the metric evaluation expects.
    - metric - evaluation metric (eg. accuracy). If metric is not defined in the pipeline,
      evaluate() function of the dataset will be called.

    parameters for calibration_dataset and input_dataset
    - path: folder containing images
    - split: provide a .txt file containing two entries in each line
      first entry in each line is image file name (starting from path above),
      for classification, second entry is class id (just set to 0 if you don't know what it is)
        example:
          image10.jpg 0
          tomato/image2.jpg 9
      for segmentation, second entry is the label image.
      for detection, second entry is not used right now in this script.
    '''

    # get dataset loaders
    imagecls_calib_dataset, imagecls_val_dataset = get_imagecls_dataset_loaders(settings)
    imageseg_calib_dataset, imageseg_val_dataset = get_imageseg_dataset_loaders(settings)
    imagedet_calib_dataset, imagedet_val_dataset = get_imagedet_dataset_loaders(settings)

    # in these examples, the session types cfgs are hardcoded for simplicity
    # however, in the configs in the root of this repository, they depend on session_type_dict
    runtime_options_tvmdlr = settings.get_runtime_options(constants.SESSION_NAME_TVMDLR, is_qat=False)
    runtime_options_tflitert = settings.get_runtime_options(constants.SESSION_NAME_TFLITERT, is_qat=False)
    runtime_options_onnxrt = settings.get_runtime_options(constants.SESSION_NAME_ONNXRT, is_qat=False)

    preproc_transforms = preprocess.PreProcessTransforms(settings)
    postproc_transforms = postprocess.PostProcessTransforms(settings)

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

    pipeline_configs = {
        'imagecls-1': dict(
            task_type='classification',
            calibration_dataset=imagecls_calib_dataset,
            input_dataset=imagecls_val_dataset,
            preprocess=preproc_transforms.get_transform_onnx(),
            session=sessions.ONNXRTSession(**onnx_session_cfg,
                runtime_options=runtime_options_onnxrt,
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx'),
            postprocess=postproc_transforms.get_transform_classification(),
            model_info=dict(metric_reference={'accuracy_top1%':71.88})
        ),
        'imagecls-2': dict(
            task_type='classification',
            calibration_dataset=imagecls_calib_dataset,
            input_dataset=imagecls_val_dataset,
            preprocess=preproc_transforms.get_transform_tflite(),
            session=sessions.TFLiteRTSession(**tflite_session_cfg,
                runtime_options=runtime_options_tflitert,
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v2_1.0_224.tflite'),
            postprocess=postproc_transforms.get_transform_classification(),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':71.9})
        ),
        'imageseg-3': dict(
            task_type='segmentation',
            calibration_dataset=imageseg_calib_dataset,
            input_dataset=imageseg_val_dataset,
            preprocess=preproc_transforms.get_transform_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_LINEAR),
            session=sessions.ONNXRTSession(**jai_session_cfg,
                runtime_options=runtime_options_onnxrt,
                model_path=f'{settings.models_path}/vision/segmentation/cocoseg21/edgeai-tv/deeplabv3lite_mobilenetv2_cocoseg21_512x512_20210405.onnx'),
            postprocess=postproc_transforms.get_transform_segmentation_onnx(),
            model_info=dict(metric_reference={'accuracy_mean_iou%':57.77})
        ),
        'imagedet-4': dict(
            task_type='detection',
            calibration_dataset=imagedet_calib_dataset,
            input_dataset=imagedet_val_dataset,
            preprocess=preproc_transforms.get_transform_tflite((300,300), (300,300), backend='cv2'),
            session=sessions.TFLiteRTSession(**tflite_session_cfg,
                runtime_options=runtime_options_tflitert,
                model_path=f'{settings.models_path}/vision/detection/coco/mlperf/ssd_mobilenet_v1_coco_20180128.tflite'),
            postprocess=postproc_transforms.get_transform_detection_tflite(),
            metric=dict(label_offset_pred=datasets.coco_det_label_offset_90to90()),
            model_info=dict(metric_reference={'accuracy_ap[.5:.95]%':23.0})
        ),
    }
    return pipeline_configs


if __name__ == '__main__':
    # the cwd must be the root of the respository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #

    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', type=str)
    cmds = parser.parse_args()
    settings = config_settings.ConfigSettings(cmds.settings_file, model_shortlist=None)

    work_dir = os.path.join(settings.modelartifacts_path, f'{settings.tensor_bits}bits')
    print(f'work_dir = {work_dir}')

    # now run the actual pipeline
    pipeline_configs = create_configs(settings, work_dir)

    # run the accuracy pipeline
    tools.run_accuracy(settings, work_dir, pipeline_configs)
