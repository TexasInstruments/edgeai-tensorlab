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

from .coco_det import *
from .coco_seg import *
from .imagenet import *
from .imagenetv2 import *
from .cityscapes import *
from .ade20k import *
from .voc_seg import *


dataset_info_dict = {
             #------------------------image classification datasets--------------------------#
             # Original ImageNet
             'imagenet':{'task_type':'classification', 'category':'imagenet', 'type':ImageNetCls, 'size':50000, 'split':'val'},
             'imagenetv1':{'task_type':'classification', 'category':'imagenet', 'type':ImageNetCls, 'size':50000, 'split':'val'},
             # ImageNetV2 as explained in imagenet_v2.py
             'imagenetv2c':{'task_type':'classification', 'category':'imagenet', 'type':ImageNetV2C, 'size':10000, 'split':'val'},
             'imagenetv2b':{'task_type':'classification', 'category':'imagenet', 'type':ImageNetV2B, 'size':10000, 'split':'val'},
             'imagenetv2a':{'task_type':'classification', 'category':'imagenet', 'type':ImageNetV2A, 'size':10000, 'split':'val'},
             # smaller versions of the original ImageNet
             'tiny-imagenet200':{'task_type':'classification', 'category':'imagenet', 'type':TinyImageNet200Cls, 'size':10000, 'split':'val'},
             'imagenet-dogs120':{'task_type':'classification', 'category':'imagenet', 'type':ImageNetDogs120Cls, 'size':20580, 'split':'train'},
             'imagenet-pseudo120':{'task_type':'classification', 'category':'imagenet', 'type':ImageNetPseudo120Cls, 'size':20580, 'split':'train'},
             'imagenet-resized-64x64':{'task_type':'classification', 'category':'imagenet', 'type':ImageNetResized64x64Cls, 'size':50000, 'split':'val'},
             #------------------------object detection datasets--------------------------#
             'coco': {'task_type':'detection', 'category':'coco', 'type':COCODetection, 'size':5000, 'split':'val2017'},
             #------------------------semantic segmentation datasets--------------------------#
             'cocoseg21': {'task_type':'segmentation', 'category':'cocoseg21', 'type':COCOSegmentation, 'size':5000, 'split':'val2017'},
             'ade20k32': {'task_type':'segmentation', 'category':'ade20k32', 'type':ADE20KSegmentation, 'size':2000, 'split':'validation'},
             'ade20k': {'task_type':'segmentation', 'category':'ade20k', 'type':ADE20KSegmentation, 'size':2000, 'split':'validation'},
             'voc2012': {'task_type':'segmentation', 'category':'voc2012', 'type':VOC2012Segmentation, 'size':1449, 'split':'val'},
             }


dataset_info_dict_experimental = {
             #------------------------semantic segmentation datasets--------------------------#
             'cityscapes': {'task_type':'segmentation', 'category':'cityscapes', 'type':CityscapesSegmentation, 'size':500, 'split':'val'},
             }


def get_dataset_info_dict(settings):
    dset_info_dict = dataset_info_dict.copy()
    if settings.experimental_models:
        dset_info_dict.update(dataset_info_dict_experimental)
    #
    return dset_info_dict
