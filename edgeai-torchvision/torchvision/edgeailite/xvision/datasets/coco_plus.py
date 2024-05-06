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
# Also includes parts from: https://github.com/pytorch/vision
# License: License: https://github.com/pytorch/vision/blob/master/LICENSE

# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
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

"""
Reference:

Microsoft COCO: Common Objects in Context,
Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays,
Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár,
https://arxiv.org/abs/1405.0312, https://cocodataset.org/
"""

import numpy as np
from torchvision.datasets.coco import COCOSegmentation
from torchvision.edgeailite import xnn

__all__ = ['coco_segmentation', 'coco_seg21']


class COCOSegmentationPlus(COCOSegmentation):
    NUM_CLASSES = 80
    def __init__(self, *args, num_classes=NUM_CLASSES, transforms=None, **kwargs):
        # 21 class is a special case, otherwise use all the classes
        # in get_item a modulo is done to map the target to the required num_classes
        super().__init__(*args, num_classes=(num_classes if num_classes==21 else self.NUM_CLASSES), **kwargs)
        self.num_classes_ = num_classes
        self.void_classes = []
        self.valid_classes = range(self.num_classes_)
        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.num_classes_)))
        self.colors = xnn.utils.get_color_palette(num_classes)
        self.colors = (self.colors * self.num_classes_)[:self.num_classes_]
        self.label_colours = dict(zip(range(self.num_classes_), self.colors))
        self.transforms = transforms

    def __getitem__(self, item):
        image, target = super().__getitem__(item)
        target = np.remainder(target, self.num_classes_)
        image = [image]
        target = [target]
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        #
        return image, target

    def num_classes(self):
        nc = []
        nc.append(self.num_classes_)
        return nc

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.num_classes_):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]
        #
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask


###########################################
# config settings
def get_config():
    dataset_config = xnn.utils.ConfigNode()
    dataset_config.num_classes = 80
    return dataset_config

def coco_segmentation(dataset_config, root, split=None, transforms=None, *args, **kwargs):
    dataset_config = get_config().merge_from(dataset_config)
    train_split = val_split = None
    split = ['train2017', 'val2017']
    for split_name in split:
        if split_name.startswith('train'):
            train_split = COCOSegmentationPlus(root, split_name, num_classes=dataset_config.num_classes,
                            transforms=transforms[0], *args, **kwargs)
        elif split_name.startswith('val'):
            val_split = COCOSegmentationPlus(root, split_name, num_classes=dataset_config.num_classes,
                            transforms=transforms[1], *args, **kwargs)
        else:
            assert False, 'unknown split'
        #
    #
    return train_split, val_split


def coco_seg21(dataset_config, root, split=None, transforms=None, *args, **kwargs):
    dataset_config = get_config().merge_from(dataset_config)
    dataset_config.num_classes = 21
    return coco_segmentation(dataset_config, root, split=split, transforms=transforms, *args, **kwargs)


