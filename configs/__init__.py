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
from jai_benchmark import utils
from jai_benchmark import datasets
from jai_benchmark import pipelines
from . import classification
from . import detection
from . import segmentation


def get_datasets(settings, download=False):
    dataset_names = ['imagenet', 'cityscapes', 'coco', 'ade20k', 'ade20k32', 'voc2012', 'coco_seg21']
    dataset_cache = {ds_name:{'calibration_dataset':None, 'input_dataset':None} for ds_name in dataset_names}

    if in_dataset_loading(settings, 'imagenet'):
        dataset_variant = settings.dataset_type_dict['imagenet'] if \
            settings.dataset_type_dict is not None else 'imagenet'
        # dataset settings
        imagenet_dataset_dict = datasets.dataset_info_dict[dataset_variant]
        ImageNetDataSetType = imagenet_dataset_dict['type']
        imagenet_split = imagenet_dataset_dict['split']
        num_imgs = imagenet_dataset_dict['size']
        # the cfg to be used to construct the dataset class
        imagenet_cls_calib_cfg = dict(
            path=f'{settings.datasets_path}/{dataset_variant}/{imagenet_split}',
            split=f'{settings.datasets_path}/{dataset_variant}/{imagenet_split}.txt',
            shuffle=True,
            num_frames=settings.quantization_params.get_calibration_frames(),
            name=dataset_variant)
        imagenet_cls_val_cfg = dict(
            path=f'{settings.datasets_path}/{dataset_variant}/{imagenet_split}',
            split=f'{settings.datasets_path}/{dataset_variant}/{imagenet_split}.txt',
            shuffle=True,
            num_frames=min(settings.num_frames,num_imgs),
            name=dataset_variant)
        # what is provided is mechanism to select one of the imagenet variants
        # but only one is selected and assigned to the key imagenet
        # all the imagenet models will use this variant.
        dataset_cache['imagenet']['calibration_dataset'] = ImageNetDataSetType(**imagenet_cls_calib_cfg, download=download)
        dataset_cache['imagenet']['input_dataset'] = ImageNetDataSetType(**imagenet_cls_val_cfg, download=False)
    #
    if in_dataset_loading(settings, 'coco'):
        coco_det_calib_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=True,
            num_frames=settings.quantization_params.get_calibration_frames(),
            name='coco')
        coco_det_val_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=False, #TODO: need to make COCODetection.evaluate() work with shuffle
            num_frames=min(settings.num_frames,5000),
            name='coco')
        dataset_cache['coco']['calibration_dataset'] = datasets.COCODetection(**coco_det_calib_cfg, download=download)
        dataset_cache['coco']['input_dataset'] = datasets.COCODetection(**coco_det_val_cfg, download=False)
    #
    if in_dataset_loading(settings, 'cocoseg21'):
        coco_seg21_calib_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=True,
            num_frames=settings.quantization_params.get_calibration_frames(),
            name='cocoseg21')
        coco_seg21_val_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=True,
            num_frames=min(settings.num_frames,5000),
            name='cocoseg21')
        dataset_cache['coco_seg21']['calibration_dataset'] = datasets.COCOSegmentation(**coco_seg21_calib_cfg, download=download)
        dataset_cache['coco_seg21']['input_dataset'] = datasets.COCOSegmentation(**coco_seg21_val_cfg, download=False)
    #
    if in_dataset_loading(settings, 'cityscapes'):
        cityscapes_seg_calib_cfg = dict(
            path=f'{settings.datasets_path}/cityscapes',
            split='val',
            shuffle=True,
            num_frames=settings.quantization_params.get_calibration_frames(),
            name='cityscapes')
        cityscapes_seg_val_cfg = dict(
            path=f'{settings.datasets_path}/cityscapes',
            split='val',
            shuffle=True,
            num_frames=min(settings.num_frames,500),
            name='cityscapes')
        dataset_cache['cityscapes']['calibration_dataset'] = datasets.CityscapesSegmentation(**cityscapes_seg_calib_cfg, download=False)
        dataset_cache['cityscapes']['input_dataset'] = datasets.CityscapesSegmentation(**cityscapes_seg_val_cfg, download=False)
    #
    if in_dataset_loading(settings, 'ade20k'):
        ade20k_seg_calib_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=settings.quantization_params.get_calibration_frames(),
            name='ade20k')
        ade20k_seg_val_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=min(settings.num_frames, 2000),
            name='ade20k')
        dataset_cache['ade20k']['calibration_dataset'] = datasets.ADE20KSegmentation(**ade20k_seg_calib_cfg, download=download)
        dataset_cache['ade20k']['input_dataset'] = datasets.ADE20KSegmentation(**ade20k_seg_val_cfg, download=False)
    #
    if in_dataset_loading(settings, 'ade20k32'):
        ade20k_seg_calib_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=settings.quantization_params.get_calibration_frames(),
            name='ade20k32')
        ade20k_seg_val_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=min(settings.num_frames, 2000),
            name='ade20k32')
        dataset_cache['ade20k32']['calibration_dataset'] = datasets.ADE20KSegmentation(**ade20k_seg_calib_cfg, num_classes=32, download=download)
        dataset_cache['ade20k32']['input_dataset'] = datasets.ADE20KSegmentation(**ade20k_seg_val_cfg, num_classes=32, download=False)
    #
    if in_dataset_loading(settings, 'voc2012'):
        voc_seg_calib_cfg = dict(
            path=f'{settings.datasets_path}/VOCdevkit/VOC2012',
            split='val',
            shuffle=True,
            num_frames=settings.quantization_params.get_calibration_frames(),
            name='voc2012')
        voc_seg_val_cfg = dict(
            path=f'{settings.datasets_path}/VOCdevkit/VOC2012',
            split='val',
            shuffle=True,
            num_frames=min(settings.num_frames, 1449),
            name='voc2012')
        dataset_cache['voc2012']['calibration_dataset'] = datasets.VOC2012Segmentation(**voc_seg_calib_cfg, download=download)
        dataset_cache['voc2012']['input_dataset'] = datasets.VOC2012Segmentation(**voc_seg_val_cfg, download=False)
    #
    return dataset_cache



def get_configs(settings, work_dir):
    # load the datasets - it is done only once and re-used for all configs
    if settings.dataset_cache is None:
        settings.dataset_cache = get_datasets(settings)
    #

    pipeline_configs = {}
    # merge all the config dictionaries
    pipeline_configs.update(classification.get_configs(settings, work_dir))
    pipeline_configs.update(detection.get_configs(settings, work_dir))
    pipeline_configs.update(segmentation.get_configs(settings, work_dir))
    if settings.experimental_models:
        from . import classification_experimental
        from . import detection_experimental
        from . import segmentation_experimental
        # now get the experimental configs
        pipeline_configs.update(classification_experimental.get_configs(settings, work_dir))
        pipeline_configs.update(detection_experimental.get_configs(settings, work_dir))
        pipeline_configs.update(segmentation_experimental.get_configs(settings, work_dir))
    #
    return pipeline_configs


def select_configs(settings, work_dir):
    pipeline_configs = get_configs(settings, work_dir)
    pipeline_runner = pipelines.PipelineRunner(pipeline_configs)
    return pipeline_runner.pipeline_configs


def download_datasets(settings, download=True):
    # just creating the dataset classes with download=True will check of the dataset folders are present
    # if the dataset folders are missing, it will be downloaded and extracted
    # set download='always' to force re-download teh datasets
    settings.dataset_cache = get_datasets(settings, download=download)
    return True


def is_all_data_loading(settings):
    return settings.dataset_loading is True or settings.dataset_loading is None


def in_dataset_loading(settings, dataset_names):
    if settings.dataset_loading is False:
        return False
    #
    if settings.dataset_loading is True or settings.dataset_loading is None:
        return True
    #
    dataset_loading = utils.as_list(settings.dataset_loading)
    dataset_names = utils.as_list(dataset_names)
    for dataset_name in dataset_names:
        if dataset_name in dataset_loading:
            return True
        #
    #
    return False


def print_all_configs(pipeline_configs=None, enable_print=False):
    if enable_print:
        for k, v in sorted(pipeline_configs.items()):
            model_name = k + '-' + v['session'].kwargs['session_name']
            print("'{}',".format(model_name))
    return




