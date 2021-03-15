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
    if settings.in_dataset_loading('imagenet'):
        dataset_cache['imagenet']['calibration_dataset'] = datasets.ImageNetCls(**settings.imagenet_cls_calib_cfg, download=download)
        dataset_cache['imagenet']['input_dataset'] = datasets.ImageNetCls(**settings.imagenet_cls_val_cfg, download=download)
    #
    if settings.in_dataset_loading('coco'):
        dataset_cache['coco']['calibration_dataset'] = datasets.COCODetection(**settings.coco_det_calib_cfg, download=download)
        dataset_cache['coco']['input_dataset'] = datasets.COCODetection(**settings.coco_det_val_cfg, download=download)
    #
    if settings.in_dataset_loading('cityscapes'):
        dataset_cache['cityscapes']['calibration_dataset'] = datasets.CityscapesSegmentation(**settings.cityscapes_seg_calib_cfg, download=False)
        dataset_cache['cityscapes']['input_dataset'] = datasets.CityscapesSegmentation(**settings.cityscapes_seg_val_cfg, download=False)
    #
    if settings.in_dataset_loading('ade20k'):
        dataset_cache['ade20k']['calibration_dataset'] = datasets.ADE20KSegmentation(**settings.ade20k_seg_calib_cfg, download=download)
        dataset_cache['ade20k']['input_dataset'] = datasets.ADE20KSegmentation(**settings.ade20k_seg_val_cfg, download=download)
    #
    if settings.in_dataset_loading('ade20k_class32'):
        dataset_cache['ade20k_class32']['calibration_dataset'] = datasets.ADE20KSegmentation(**settings.ade20k_seg_calib_cfg, num_classes=32, download=download)
        dataset_cache['ade20k_class32']['input_dataset'] = datasets.ADE20KSegmentation(**settings.ade20k_seg_val_cfg, num_classes=32, download=download)
    #
    if settings.in_dataset_loading('voc2012'):
        dataset_cache['voc2012']['calibration_dataset'] = datasets.VOC2012Segmentation(**settings.voc_seg_calib_cfg, download=download)
        dataset_cache['voc2012']['input_dataset'] = datasets.VOC2012Segmentation(**settings.voc_seg_val_cfg, download=download)
    #
    return dataset_cache


def download_datasets(settings):
    # just creating the dataset classes with download=True will check of the dataset folders are present
    # if the dataset folders are missing, it will be downloaded and extracted
    settings.dataset_cache = get_datasets(settings, download=True)
    return True

def print_all_configs(pipeline_configs=None, enable_print=False):
    if enable_print:
        for k, v in sorted(pipeline_configs.items()):
            v['session'].kwargs['session_name']
            print(k + '-' + v['session'].kwargs['session_name'])
    return

def get_configs(settings, work_dir):
    # load the datasets - it is done only once and re-used for all configs
    if settings.dataset_cache is None:
        settings.dataset_cache = get_datasets(settings)
    #

    pipeline_configs = {}
    # merge all the config dictionaries
    for task in utils.as_list(settings.task_selection):
        if task in ('classification', None):
            pipeline_configs.update(classification.get_configs(settings, work_dir))
        #
        if task in ('detection', None):
            pipeline_configs.update(detection.get_configs(settings, work_dir))
        #
        if task in ('segmentation', None):
            pipeline_configs.update(segmentation.get_configs(settings, work_dir))
        #
    #
    return pipeline_configs
