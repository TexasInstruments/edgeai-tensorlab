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
from .. import utils
from .. import datasets
from . import classification
from . import segmentation
from . import detection


def get_datasets(settings, download=False):
    dataset_cache = {'imagenet':{'calibration_dataset':None, 'input_dataset':None},
                     'cityscapes':{'calibration_dataset':None, 'input_dataset':None},
                     'coco':{'calibration_dataset':None, 'input_dataset':None},
                     'ade20k':{'calibration_dataset':None, 'input_dataset':None},
                     'ade20k_class32':{'calibration_dataset':None, 'input_dataset':None},
                     'voc2012':{'calibration_dataset':None, 'input_dataset':None},
                     }

    if in_dataset_loading(settings, 'imagenet'):
        # dataset settings
        imagenet_cls_calib_cfg = dict(
            path=f'{settings.datasets_path}/imagenet/val',
            split=f'{settings.datasets_path}/imagenet/val.txt',
            shuffle=True,
            num_frames=settings.quantization_params.get_num_frames_calib())
        imagenet_cls_val_cfg = dict(
            path=f'{settings.datasets_path}/imagenet/val',
            split=f'{settings.datasets_path}/imagenet/val.txt',
            shuffle=True,
            num_frames=min(settings.num_frames,50000))
        dataset_cache['imagenet']['calibration_dataset'] = datasets.ImageNetCls(**imagenet_cls_calib_cfg, download=download)
        dataset_cache['imagenet']['input_dataset'] = datasets.ImageNetCls(**imagenet_cls_val_cfg, download=download)
    #
    if in_dataset_loading(settings, 'coco'):
        coco_det_calib_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=True,
            num_frames=settings.quantization_params.get_num_frames_calib())
        coco_det_val_cfg = dict(
            path=f'{settings.datasets_path}/coco',
            split='val2017',
            shuffle=False, #TODO: need to make COCODetection.evaluate() work with shuffle
            num_frames=min(settings.num_frames,5000))
        dataset_cache['coco']['calibration_dataset'] = datasets.COCODetection(**coco_det_calib_cfg, download=download)
        dataset_cache['coco']['input_dataset'] = datasets.COCODetection(**coco_det_val_cfg, download=download)
    #
    if in_dataset_loading(settings, 'cityscapes'):
        cityscapes_seg_calib_cfg = dict(
            path=f'{settings.datasets_path}/cityscapes',
            split='val',
            shuffle=True,
            num_frames=settings.quantization_params.get_num_frames_calib())

        cityscapes_seg_val_cfg = dict(
            path=f'{settings.datasets_path}/cityscapes',
            split='val',
            shuffle=True,
            num_frames=min(settings.num_frames,500))
        dataset_cache['cityscapes']['calibration_dataset'] = datasets.CityscapesSegmentation(**cityscapes_seg_calib_cfg, download=False)
        dataset_cache['cityscapes']['input_dataset'] = datasets.CityscapesSegmentation(**cityscapes_seg_val_cfg, download=False)
    #
    if in_dataset_loading(settings, 'ade20k'):
        ade20k_seg_calib_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=settings.quantization_params.get_num_frames_calib())

        ade20k_seg_val_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=min(settings.num_frames, 2000))
        dataset_cache['ade20k']['calibration_dataset'] = datasets.ADE20KSegmentation(**ade20k_seg_calib_cfg, download=download)
        dataset_cache['ade20k']['input_dataset'] = datasets.ADE20KSegmentation(**ade20k_seg_val_cfg, download=download)
    #
    if in_dataset_loading(settings, 'ade20k_class32'):
        ade20k_seg_calib_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=settings.quantization_params.get_num_frames_calib())

        ade20k_seg_val_cfg = dict(
            path=f'{settings.datasets_path}/ADEChallengeData2016',
            split='validation',
            shuffle=True,
            num_frames=min(settings.num_frames, 2000))
        dataset_cache['ade20k_class32']['calibration_dataset'] = datasets.ADE20KSegmentation(**ade20k_seg_calib_cfg, num_classes=32, download=download)
        dataset_cache['ade20k_class32']['input_dataset'] = datasets.ADE20KSegmentation(**ade20k_seg_val_cfg, num_classes=32, download=download)
    #
    if in_dataset_loading(settings, 'voc2012'):
        voc_seg_calib_cfg = dict(
            path=f'{settings.datasets_path}/VOCdevkit/VOC2012',
            split='val',
            shuffle=True,
            num_frames=settings.quantization_params.get_num_frames_calib())
        voc_seg_val_cfg = dict(
            path=f'{settings.datasets_path}/VOCdevkit/VOC2012',
            split='val',
            shuffle=True,
            num_frames=min(settings.num_frames, 1449))
        dataset_cache['voc2012']['calibration_dataset'] = datasets.VOC2012Segmentation(**voc_seg_calib_cfg, download=download)
        dataset_cache['voc2012']['input_dataset'] = datasets.VOC2012Segmentation(**voc_seg_val_cfg, download=download)
    #
    return dataset_cache


def in_dataset_loading(settings, dataset_name):
    if settings.dataset_loading is False:
        return False
    #
    if settings.dataset_loading is True or settings.dataset_loading is None:
        return True
    #
    dataset_loading = utils.as_list(settings.dataset_loading)
    if dataset_name in dataset_loading:
        return True
    #
    return False


def download_datasets(settings):
    # just creating the dataset classes with download=True will check of the dataset folders are present
    # if the dataset folders are missing, it will be downloaded and extracted
    settings.dataset_cache = get_datasets(settings, download=True)
    return True

def print_all_configs(pipeline_configs=None, enable_print=False):
    if enable_print:
        for k, v in sorted(pipeline_configs.items()):
            v['session'].kwargs['session_name']
            model_name = k + '-' + v['session'].kwargs['session_name']
            print("'{}',".format(model_name))
    return

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
    return pipeline_configs
