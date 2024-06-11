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
import tempfile
from colorama import Fore
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import json

from .. import utils
from .dataset_base import *

__all__ = ['COCOSegmentation']


class COCOSegmentation(DatasetBase):
    def __init__(self, num_classes=21, download=False, num_frames=None, name="cocoseg21", **kwargs):
        super().__init__(num_classes=num_classes, num_frames=num_frames, name=name, **kwargs)
        self.force_download = True if download == 'always' else False
        assert 'path' in self.kwargs and 'split' in self.kwargs, 'kwargs must have path and split'
        path = self.kwargs['path']
        split = self.kwargs['split']
        root = path
        if download:
            self.download(path, split)
        #
        self.kwargs['num_frames'] = self.kwargs.get('num_frames', None)
        self.name = name
        self.tempfiles = []

        self.num_classes = 80 if num_classes is None else num_classes
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

        shuffle = self.kwargs.get('shuffle', False)
        image_base_dir = 'images' if ('images' in dataset_folders) else ''
        image_base_dir = os.path.join(root, image_base_dir)
        image_split_dirs = os.listdir(image_base_dir)
        self.image_dir = os.path.join(image_base_dir, split)

        self.annotation_file = os.path.join(annotations_dir, f'instances_{split}.json')
        self.coco_dataset = COCO(self.annotation_file)

        self.cat_ids = self.coco_dataset.getCatIds()
        img_ids = self.coco_dataset.getImgIds()
        self.img_ids = self._remove_images_without_annotations(img_ids)

        max_frames = len(self.coco_dataset.imgs)
        num_frames = self.kwargs.get('num_frames', None)
        num_frames = min(num_frames, max_frames) if num_frames is not None else max_frames

        imgs_list = list(self.coco_dataset.imgs.items())
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(imgs_list)
        #
        self.coco_dataset.imgs = {k:v for k,v in imgs_list[:num_frames]}

        max_frames = len(self.coco_dataset.imgs)
        num_frames = self.kwargs.get('num_frames', None)
        num_frames = min(num_frames, max_frames) if num_frames is not None else max_frames

        self.cat_ids = self.coco_dataset.getCatIds()
        self.img_ids = self.coco_dataset.getImgIds()
        self.num_frames = self.kwargs['num_frames'] = num_frames

        run_dir = self.kwargs.get('run_dir', None)
        if run_dir is None:
            temp_dir = tempfile.TemporaryDirectory()
            run_dir = temp_dir.name
            self.tempfiles.append(temp_dir)
        #
        self.label_dir = os.path.join(run_dir, 'labels')
        with open(self.annotation_file) as afp:
            self.dataset_store = json.load(afp)
        #
        self.kwargs['dataset_info'] = self.get_dataset_info()

    def download(self, path, split):
        root = path
        images_folder = os.path.join(path, split)
        annotations_folder = os.path.join(path, 'annotations')
        if (not self.force_download) and os.path.exists(path) and \
                os.path.exists(images_folder) and os.path.exists(annotations_folder):
            print(utils.log_color('\nINFO', 'dataset exists - will reuse', path))
            return
        #
        print(utils.log_color('\nINFO', 'downloading and preparing dataset', path + ' This may take some time.'))
        print(f'{Fore.YELLOW}'
              f'\nCOCO Dataset:'
              f'\n    Microsoft COCO: Common Objects in Context, '
              f'\n        Tsung-Yi Lin, et.al. https://arxiv.org/abs/1405.0312\n'
              f'\n    Visit the following url to know more about the COCO dataset. '
              f'\n        https://cocodataset.org/ '
              f'{Fore.RESET}\n')

        dataset_url = 'http://images.cocodataset.org/zips/val2017.zip'
        extra_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        download_root = os.path.join(root, 'download')
        dataset_path = utils.download_file(dataset_url, root=download_root, extract_root=root)
        extra_path = utils.download_file(extra_url, root=download_root, extract_root=root)
        print(utils.log_color('\nINFO', 'dataset ready', path))
        return

    def _get_root(self, path):
        path = path.rstrip('/')
        root = os.sep.join(os.path.split(path)[:-1])
        return root

    def __getitem__(self, idx, with_label=False):
        img_id = self.img_ids[idx]
        img = self.coco_dataset.loadImgs([img_id])[0]
        image_path = os.path.join(self.image_dir, img['file_name'])
        if with_label:
            os.makedirs(self.label_dir, exist_ok=True)
            ann_ids = self.coco_dataset.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco_dataset.loadAnns(ann_ids)
            image = PIL.Image.open(image_path)
            image, anno = self._filter_and_remap_categories(image, anno)
            image, target = self._convert_polys_to_mask(image, anno)
            # write the label file to a temorary dir so that it can be used by evaluate()
            image_basename = os.path.basename(image_path)
            label_path = os.path.join(self.label_dir, image_basename)
            label_path = os.path.splitext(label_path)[0] + '.png'
            cv2.imwrite(label_path, target)
            return image_path, label_path
        else:
            return image_path
        #

    def __len__(self):
        return self.num_frames

    def __del__(self):
        for t in self.tempfiles:
            t.cleanup()
        #

    def encode_segmap(self, label_img):
        # label has already been encoded
        return label_img

    def __call__(self, predictions, **kwargs):
        return self.evaluate(predictions, **kwargs)

    def evaluate(self, predictions, **kwargs):
        cmatrix = None
        num_frames = min(self.num_frames, len(predictions))
        for n in range(num_frames):
            image_file, label_file = self.__getitem__(n, with_label=True)
            label_img = PIL.Image.open(label_file)
            label_img = self.encode_segmap(label_img)
            # reshape prediction is needed
            output = predictions[n]
            output = output.astype(np.uint8)
            output = output[0] if (output.ndim > 2 and output.shape[0] == 1) else output
            output = output[:2] if (output.ndim > 2 and output.shape[2] == 1) else output
            # compute metric
            cmatrix = utils.confusion_matrix(cmatrix, output, label_img, self.num_classes)
        #
        accuracy = utils.segmentation_accuracy(cmatrix)
        return accuracy

    def get_dataset_info(self):
        if 'dataset_info' in self.kwargs:
            return self.kwargs['dataset_info']
        #
        # return only info and categories for now as the whole thing could be quite large.
        dataset_store = dict()
        for key in ('info', 'categories'):
            if key in self.dataset_store.keys():
                dataset_store.update({key: self.dataset_store[key]})
            #
        #
        dataset_store.update(dict(color_map=self.get_color_map()))        
        return dataset_store

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


if __name__ == '__main__':
    # from inside the folder jacinto_ai_benchmark, run the following:
    # python3 -m edgeai_benchmark.datasets.coco_seg
    # to create a converted dataset if you wish to load it using the dataset loader ImageSegmentation() in image_seg.py
    # to load it using CocoSegmentation dataset in this file, this conversion is not required.
    import shutil
    output_folder = './dependencies/datasets/coco-seg21-converted'
    split = 'val2017'
    coco_seg = COCOSegmentation(path='./dependencies/datasets/coco', split=split)
    num_frames = len(coco_seg)

    images_output_folder = os.path.join(output_folder, split, 'images')
    labels_output_folder = os.path.join(output_folder, split, 'labels')
    os.makedirs(images_output_folder)
    os.makedirs(labels_output_folder)

    output_filelist = os.path.join(output_folder, f'{split}.txt')
    with open(output_filelist, 'w') as list_fp:
        for n in range(num_frames):
            image_path, label_path = coco_seg.__getitem__(n, with_label=True)
            # not needed - encode_segmap doesn't do anything
            # image = PIL.Image.open(image_path)
            # label_img = PIL.Image.open(label_path)
            # label_img = coco_seg.encode_segmap(label_img)
            image_output_filename = os.path.join(images_output_folder, os.path.basename(image_path))
            label_output_filename = os.path.join(labels_output_folder, os.path.basename(label_path))
            shutil.copy2(image_path, image_output_filename)
            shutil.copy2(label_path, label_output_filename)
            list_fp.write(f'images/{os.path.basename(image_output_filename)} labels/{os.path.basename(label_output_filename)}\n')
        #
    #
