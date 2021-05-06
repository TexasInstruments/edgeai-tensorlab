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


def get_imagenetcls_dataset_loaders(settings, download=False):
    dataset_calib_cfg = dict(
        path=f'{settings.datasets_path}/imagenet/val',
        split=f'{settings.datasets_path}/imagenet/val.txt',
        shuffle=True,
        num_frames=settings.quantization_params.get_calibration_frames())

    # dataset parameters for actual inference
    dataset_val_cfg = dict(
        path=f'{settings.datasets_path}/imagenet/val',
        split=f'{settings.datasets_path}/imagenet/val.txt',
        shuffle=True,
        num_frames=min(settings.num_frames,50000))

    calib_dataset = datasets.ImageCls(**dataset_calib_cfg, download=download)
    val_dataset = datasets.ImageCls(**dataset_val_cfg, download=download)
    return calib_dataset, val_dataset


def get_cocoseg21_dataset_loaders(settings, download=False):
    dataset_calib_cfg = dict(
        path=f'{settings.datasets_path}/coco-seg21-converted/val2017',
        split=f'{settings.datasets_path}/coco-seg21-converted/val2017.txt',
        shuffle=True,
        num_classes=21,
        num_frames=settings.quantization_params.get_calibration_frames())

    # dataset parameters for actual inference
    dataset_val_cfg = dict(
        path=f'{settings.datasets_path}/coco-seg21-converted/val2017',
        split=f'{settings.datasets_path}/coco-seg21-converted/val2017.txt',
        shuffle=True,
        num_classes=21,
        num_frames=min(settings.num_frames,5000))

    calib_dataset = datasets.ImageSeg(**dataset_calib_cfg, download=download)
    val_dataset = datasets.ImageSeg(**dataset_val_cfg, download=download)
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

    dataset parameters for calibration
    - path: folder containing images
    - split: provide a .txt file containing two entries in each line
      first entry is image file name (starting from path above),
      second entry is class id (just set to 0 if you don't know what it is)
    example:
    image10.jpg 0
    tomato/image2.jpg 9
    '''

    # get dataset loaders
    imagenetcls_calib_dataset, imagenetcls_val_dataset = get_imagenetcls_dataset_loaders(settings)
    cocoseg21_calib_dataset, cocoseg21_val_dataset = get_cocoseg21_dataset_loaders(settings)

    # in these examples, the session types cfgs are hardcoded for simplicity
    # however, in the configs in the root of this repository, they depend on session_type_dict
    runtime_options_tvmdlr = settings.get_runtime_options(constants.SESSION_NAME_TVMDLR, is_qat=False)
    runtime_options_tflitert = settings.get_runtime_options(constants.SESSION_NAME_TFLITERT, is_qat=False)
    runtime_options_onnxrt = settings.get_runtime_options(constants.SESSION_NAME_ONNXRT, is_qat=False)

    pipeline_configs = {
        'imagenetcls-1': dict(
            task_type='classification',
            calibration_dataset=imagenetcls_calib_dataset,
            input_dataset=imagenetcls_val_dataset,
            preprocess=settings.get_preproc_onnx(),
            session=sessions.ONNXRTSession(
                work_dir=work_dir, target_device=settings.target_device, runtime_options=runtime_options_onnxrt,
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_opset9.onnx'),
            postprocess=settings.get_postproc_classification(),
            model_info=dict(metric_reference={'accuracy_top1%':71.88})
        ),
        'imagenetcls-2': dict(
            task_type='classification',
            calibration_dataset=imagenetcls_calib_dataset,
            input_dataset=imagenetcls_val_dataset,
            preprocess=settings.get_preproc_tflite(),
            session=sessions.TFLiteRTSession(
                work_dir=work_dir, target_device=settings.target_device, runtime_options=runtime_options_tflitert,
                model_path=f'{settings.models_path}/vision/classification/imagenet1k/tf1-models/mobilenet_v2_1.0_224.tflite'),
            postprocess=settings.get_postproc_classification(),
            metric=dict(label_offset_pred=-1),
            model_info=dict(metric_reference={'accuracy_top1%':71.9})
        ),
        'cocoseg21-3': dict(
            task_type='segmentation',
            calibration_dataset=cocoseg21_calib_dataset,
            input_dataset=cocoseg21_val_dataset,
            preprocess=settings.get_preproc_jai((512,512), (512,512), backend='cv2', interpolation=cv2.INTER_LINEAR),
            session=sessions.ONNXRTSession(
                work_dir=work_dir, target_device=settings.target_device, runtime_options=runtime_options_onnxrt,
                model_path=f'{settings.models_path}/vision/segmentation/cocoseg21/jai-pytorch/deeplabv3lite_mobilenetv2_cocoseg21_512x512_20210405.onnx'),
            postprocess=settings.get_postproc_segmentation_onnx(),
            model_info=dict(metric_reference={'accuracy_mean_iou%':57.77})
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
    parser.add_argument('--work_dirs', type=str, default='./work_dirs')
    cmds = parser.parse_args()
    settings = config_settings.ConfigSettings(cmds.settings_file, model_shortlist=None)

    expt_name = os.path.splitext(os.path.basename(__file__))[0]
    work_dir = os.path.join(cmds.work_dirs, expt_name, f'{settings.tensor_bits}bits')
    print(f'work_dir = {work_dir}')

    # now run the actual pipeline
    pipeline_configs = create_configs(settings, work_dir)

    # run the accuracy pipeline
    tools.run_accuracy(settings, work_dir, pipeline_configs)
