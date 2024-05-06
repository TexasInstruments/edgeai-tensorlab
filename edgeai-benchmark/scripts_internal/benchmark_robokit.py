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


def get_imageseg_robokit_dataset_loaders(settings, download=False):
    dataset_calib_cfg = dict(
        path=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd',
        split=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd/train_img_gt_pair.txt',
        num_classes=19,
        shuffle=True,
        num_frames=min(settings.calibration_frames,150))

    # dataset parameters for actual inference
    dataset_val_cfg = dict(
        path=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd',
        split=f'{settings.datasets_path}/ti-robokit_semseg_zed1hd/val_img_gt_pair.txt',
        num_classes=19,
        shuffle=True,
        num_frames=min(settings.num_frames,49))

    calib_dataset = datasets.ImageSegmentation(**dataset_calib_cfg, download=download)
    val_dataset = datasets.ImageSegmentation(**dataset_val_cfg, download=download)
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
    imageseg_robokit_calib_dataset, imageseg_robokit_val_dataset = get_imageseg_robokit_dataset_loaders(settings)

    # in these examples, the session types cfgs are hardcoded for simplicity
    # however, in the configs in the root of this repository, they depend on session_type_dict
    # runtime_options_tflitert = settings.get_runtime_options(constants.SESSION_NAME_TFLITERT, is_qat=False)
    # runtime_options_onnxrt = settings.get_runtime_options(constants.SESSION_NAME_ONNXRT, is_qat=False)
    runtime_options_tvmdlr = settings.get_runtime_options(constants.SESSION_NAME_TVMDLR, is_qat=False)
    runtime_options_tvmdlr_qat = settings.get_runtime_options(constants.SESSION_NAME_TVMDLR, is_qat=True)

    preproc_transforms = preprocess.PreProcessTransforms(settings)
    postproc_transforms = postprocess.PostProcessTransforms(settings)

    jai_session_cfg = sessions.get_jai_session_cfg(work_dir=work_dir, target_device=settings.target_device,
                            tidl_offload=settings.tidl_offload, input_optimization=settings.input_optimization)

    pipeline_configs = {
        'ss-robokit1-qat': dict(
            task_type='segmentation',
            calibration_dataset=imageseg_robokit_calib_dataset,
            input_dataset=imageseg_robokit_val_dataset,
            preprocess=preproc_transforms.get_transform_jai((432,768), (432,768), backend='cv2', interpolation=cv2.INTER_AREA),
            session=sessions.TVMDLRSession(**jai_session_cfg,
                runtime_options=runtime_options_tvmdlr_qat,
                model_path=f'{settings.models_path}/vision/segmentation/ti-robokit/edgeai-tv/robokit-zed1hd_deeplabv3lite_mobilenetv2_tv_768x432_qat-p2.onnx'),
            postprocess=postproc_transforms.get_transform_segmentation_onnx(),
            model_info=dict(metric_reference={'accuracy_mean_iou%':None})
        ),
        'ss-robokit2': dict(
            task_type='segmentation',
            calibration_dataset=imageseg_robokit_calib_dataset,
            input_dataset=imageseg_robokit_val_dataset,
            preprocess=preproc_transforms.get_transform_jai((432,768), (432,768), backend='cv2', interpolation=cv2.INTER_AREA),
            session=sessions.TVMDLRSession(**jai_session_cfg,
                runtime_options=runtime_options_tvmdlr,
                model_path=f'{settings.models_path}/vision/segmentation/ti-robokit/edgeai-tv/robokit-zed1hd_deeplabv3lite_mobilenetv2_tv_768x432.onnx'),
            postprocess=postproc_transforms.get_transform_segmentation_onnx(),
            model_info=dict(metric_reference={'accuracy_mean_iou%':None})
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
    settings = config_settings.ConfigSettings(cmds.settings_file, model_shortlist=None,
                                              models_path='../edgeai-modelforest/models')

    work_dir = os.path.join(settings.modelartifacts_path, f'{settings.tensor_bits}bits')
    print(f'work_dir = {work_dir}')

    # now run the actual pipeline
    pipeline_configs = create_configs(settings, work_dir)

    # run the accuracy pipeline
    tools.run_accuracy(settings, work_dir, pipeline_configs)
