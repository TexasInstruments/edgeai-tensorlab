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
from PIL import Image
import os
import random
import numpy as np
import copy
import cv2

from edgeai_torchmodelopt import xnn

__all__ = ['coco_segmentation', 'coco_seg21']


class COCOSegmentation():
    '''
    Modified from torchvision: https://github.com/pytorch/vision/references/segmentation/coco_utils.py
    Reference: https://github.com/pytorch/vision/blob/master/docs/source/models.rst
    '''
    def __init__(self, root, split, shuffle=False, num_imgs=None, num_classes=None):
        from pycocotools.coco import COCO
        num_classes = 80 if num_classes is None else num_classes
        if num_classes == 21:
            self.categories = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
            self.class_names = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                                'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        else:
            self.categories = range(num_classes)
            self.class_names = None
        #

        dataset_folders = os.listdir(root)
        assert 'annotations' in dataset_folders, 'invalid path to coco dataset annotations'
        annotations_dir = os.path.join(root, 'annotations')

        image_base_dir = 'images' if ('images' in dataset_folders) else ''
        image_base_dir = os.path.join(root, image_base_dir)
        image_split_dirs = os.listdir(image_base_dir)
        image_dir = os.path.join(image_base_dir, split)

        self.coco_dataset = COCO(os.path.join(annotations_dir, f'instances_{split}.json'))

        self.cat_ids = self.coco_dataset.getCatIds()
        img_ids = self.coco_dataset.getImgIds()
        self.img_ids = self._remove_images_without_annotations(img_ids)

        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.img_ids)
        #

        if num_imgs is not None:
            self.img_ids = self.img_ids[:num_imgs]
            self.coco_dataset.imgs = {k:self.coco_dataset.imgs[k] for k in self.img_ids}
        #

        imgs = []
        for img_id in self.img_ids:
            img = self.coco_dataset.loadImgs([img_id])[0]
            imgs.append(os.path.join(image_dir, img['file_name']))
        #
        self.imgs = imgs
        self.num_imgs = len(self.imgs)

    def __getitem__(self, idx, with_label=True):
        if with_label:
            image = Image.open(self.imgs[idx])
            ann_ids = self.coco_dataset.getAnnIds(imgIds=self.img_ids[idx], iscrowd=None)
            anno = self.coco_dataset.loadAnns(ann_ids)
            image, anno = self._filter_and_remap_categories(image, anno)
            image, target = self._convert_polys_to_mask(image, anno)
            image = np.array(image)
            if image.ndim==2 or image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            #
            target = np.array(target)
            return image, target
        else:
            return self.imgs[idx]
        #

    def __len__(self):
        return self.num_imgs

    def _remove_images_without_annotations(self, img_ids):
        ids = []
        for ds_idx, img_id in enumerate(img_ids):
            ann_ids = self.coco_dataset.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco_dataset.loadAnns(ann_ids)
            if self.categories:
                anno = [obj for obj in anno if obj["category_id"] in self.categories]
            if self._has_valid_annotation(anno):
                ids.append(img_id)
            #
        #
        return ids

    def _has_valid_annotation(self, anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if more than 1k pixels occupied in the image
        return sum(obj["area"] for obj in anno) > 1000

    def _filter_and_remap_categories(self, image, anno, remap=True):
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not remap:
            return image, anno
        #
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        #
        return image, anno

    def _convert_polys_to_mask(self, image, anno):
        w, h = image.size
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if segmentations:
            masks = self._convert_poly_to_mask(segmentations, h, w)
            cats = np.array(cats, dtype=masks.dtype)
            cats = cats.reshape(-1, 1, 1)
            # merge all instance masks into a single segmentation map
            # with its corresponding categories
            target = (masks * cats).max(axis=0)
            # discard overlapping instances
            target[masks.sum(0) > 1] = 255
        else:
            target = np.zeros((h, w), dtype=np.uint8)
        #
        return image, target

    def _convert_poly_to_mask(self, segmentations, height, width):
        from pycocotools import mask as coco_mask
        masks = []
        for polygons in segmentations:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = mask.any(axis=2)
            mask = mask.astype(np.uint8)
            masks.append(mask)
        if masks:
            masks = np.stack(masks, axis=0)
        else:
            masks = np.zeros((0, height, width), dtype=np.uint8)
        return masks


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
        self.color_map = xnn.utils.get_color_palette(num_classes)
        self.color_map = (self.color_map * self.num_classes_)[:self.num_classes_]
        self.label_colours = dict(zip(range(self.num_classes_), self.color_map))
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


