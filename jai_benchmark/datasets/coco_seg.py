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

################################################################################

# Also includes parts from: https://github.com/cocodataset/cocoapi (pycocotools)
# License: https://github.com/cocodataset/cocoapi/blob/master/license.txt

# Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the FreeBSD Project.

"""
Reference:

Microsoft COCO: Common Objects in Context,
Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays,
Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, Piotr Dollár,
https://arxiv.org/abs/1405.0312, https://cocodataset.org/
"""


import os
import random
import copy
import numpy as np
import PIL
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from .. import utils

__all__ = ['COCOSegmentation']


class COCOSegmentation(utils.ParamsBase):
    def __init__(self, inData, num_imgs=None, num_classes=21, download=False, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        assert 'path' in kwargs and 'split' in kwargs, 'kwargs must have path and split'
        path = kwargs['path']
        split = kwargs['split']
        if download:
            self.download(path, split)
        #

        self.categories = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72] \
            if num_classes == 21 else None
        #
        self.num_classes = num_classes
        assert isinstance(inData, dict) and 'path' in list(inData.keys()) and 'split' in list(inData.keys()), 'inData must be a dict'

        dataset_folders = os.listdir(inData['path'])
        assert 'annotations' in dataset_folders, 'invalid path to coco dataset annotations'
        annotations_dir = os.path.join(inData['path'], 'annotations')

        shuffle = inData['shuffle'] if (isinstance(inData, dict) and 'shuffle' in inData) else False
        image_base_dir = 'images' if ('images' in dataset_folders) else ''
        image_base_dir = os.path.join(inData['path'], image_base_dir)
        image_split_dirs = os.listdir(image_base_dir)
        assert inData['split'] in image_split_dirs, f'invalid path to coco dataset images/split {inData["split"]}'
        image_dir = os.path.join(image_base_dir, inData['split'])

        self.coco_dataset = COCO(os.path.join(annotations_dir, f'instances_{inData["split"]}.json'))

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
        # call the utils.ParamsBase.initialize()
        super().initialize()

    def download(self, path, split):
        root = path
        images_folder = os.path.join(path, split)
        annotations_folder = os.path.join(path, 'annotations')
        if os.path.exists(path) and os.path.exists(images_folder) and os.path.exists(annotations_folder):
            return
        #
        print('Important: Please visit the urls: https://cocodataset.org/#home and '
              'https://cocodataset.org/#termsofuse to understand more about the COCO dataset '
              'and accept the terms and conditions under which it can be used. ')

        dataset_url = 'http://images.cocodataset.org/zips/val2017.zip'
        extra_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        dataset_path = utils.download_file(dataset_url, root=root)
        extra_path = utils.download_file(extra_url, root=root)
        return

    def _get_root(self, path):
        path = path.rstrip('/')
        root = os.sep.join(os.path.split(path)[:-1])
        return root

    def __getitem__(self, idx, with_label=False):
        if with_label:
            image = PIL.Image.open(self.imgs[idx])
            ann_ids = self.coco_dataset.getAnnIds(imgIds=self.img_ids[idx], iscrowd=None)
            anno = self.coco_dataset.loadAnns(ann_ids)
            image, anno = self._filter_and_remap_categories(image, anno)
            image, target = self._convert_polys_to_mask(image, anno)
            return image, target
        else:
            return self.imgs[idx]
        #

    def __len__(self):
        return self.num_imgs

    def evaluate(self, outputs, **kwargs):
        cmatrix = None
        for n in range(self.num_imgs):
            image, label_img = self.__getitem__(n, with_label=True)
            gtHeight, gtWidth = label_img.shape[:2]

            output = outputs[n]
            output = output.astype(np.uint8)
            output = output[0] if (output.ndim > 2 and output.shape[0] == 1) else output
            output = output[:2] if (output.ndim > 2 and output.shape[2] == 1) else output

            #convert to pillow image - not necessary
            #output = PIL.Image.fromarray(output, mode='L') if isinstance(output, np.ndarray) else input

            resample_type = cv2.INTER_NEAREST if isinstance(output, np.ndarray) else PIL.Image.NEAREST
            output = utils.resize_pad_crop_image(output, resize_w=gtWidth, resize_h=gtHeight, inResizeType=0,
                                                 resample_type=resample_type)
            cmatrix = utils.confusion_matrix(cmatrix, output, label_img, self.num_classes)
        #

        accuracy = utils.segmentation_accuracy(cmatrix)
        return accuracy


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
    #

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

