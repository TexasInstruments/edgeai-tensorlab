#################################################################################
# Copyright (c) 2018-2021, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################

import warnings
from torchvision.datasets.cityscapes import Cityscapes
from torchvision.datasets.voc import VOCSegmentation

__all__ = ['cityscales_semantic', 'cityscales_instance', 'voc_segmentation']


#####################################################################
# target_type ['instance', 'semantic', 'polygon', 'color']
def cityscapes_base(dataset_config, root, split=('train', 'val'), transforms=None, target_type=None):
    train_split = val_split = None
    for split_name in split:
        if split_name == 'train':
            train_split = Cityscapes(root, split=split_name, mode='fine', target_type=target_type, transform=transforms[0])
        elif split_name == 'val':
            val_split = Cityscapes(root, split=split_name, mode='fine', target_type=target_type, transform=transforms[1])
        else:
            pass

    return train_split, val_split


def cityscales_semantic(dataset_config, root, split=('train', 'val'), transforms=None, target_type='semantic'):
    return cityscapes_base(dataset_config, root, split, transforms, target_type)


def cityscales_instance(dataset_config, root, split=('train', 'val'), transforms=None, target_type='instance'):
    return cityscapes_base(dataset_config, root, split, transforms, target_type)


#####################################################################
def voc_segmentation(dataset_config, root, split=None, transforms=None, target_type=None):
    split = ('trainaug_noval', 'val') if split is None else split
    warnings_str = '''
    Note: 'trainaug' set of VOC2012 Segmentation has images in 'val' as well, so validation results won't be indicative of the test results.
    To get test results when using 'trainaug' for training, submit for testing on Pascal VOC 2012 test server.
    However, when training with 'trainaug_noval', validation results are correct, and expected to be indicative of the test.
    But since 'trainaug_noval', has fewer images, the results may be be poorer. Using 'trainaug_noval' as default for now.
    For more details, see here: https://github.com/DrSleep/tensorflow-deeplab-resnet
    And here: http://home.bharathh.info/pubs/codes/SBD/download.html
    And here: https://github.com/tensorflow/models/blob/master/research/deeplab/train.py 
    The list of images for trainaug_noval is avalibale here: http://home.bharathh.info/pubs/codes/SBD/train_noval.txt
    '''
    if split[0] == 'trainaug':
        warnings.warn(warnings_str)

    train_split = val_split = None
    for split_name in split:
        if split_name == 'train' or split_name == 'trainaug' or split_name == 'trainaug_noval':
            train_split = VOCSegmentation(root, image_set=split_name, transforms=transforms[0], download=False)
        elif split_name == 'val':
            val_split = VOCSegmentation(root, image_set=split_name, transforms=transforms[1], download=False)
        else:
            pass

    return train_split, val_split

