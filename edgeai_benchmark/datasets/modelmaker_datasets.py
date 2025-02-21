# Copyright (c) 2018-2021, Texas Instruments Incorporated
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
import json
import PIL
import numpy as np
import cv2
import random
import tempfile
import copy
import platform
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from .dataset_base import *
from . import coco_det
from . import coco_seg


class ModelMakerDetectionDataset(coco_det.COCODetection):
    def __init__(self, num_classes=None, download=False, num_frames=None, name='modelmaker',
                 annotation_prefix='instances', **kwargs):
        assert 'path' in kwargs and 'split' in kwargs, 'kwargs must have path and split'
        path = kwargs['path']
        split = kwargs['split']
        image_dir = kwargs.pop('image_dir', os.path.join(path, split))
        annotation_file = kwargs.pop('annotation_file', os.path.join(path, 'annotations', f'{annotation_prefix}_{split}.json'))
        super().__init__(num_classes=num_classes, image_dir=image_dir, annotation_file=annotation_file,
                         download=False, num_frames=num_frames, name=name, **kwargs)
        with open(annotation_file) as afp:
            self.dataset_store = json.load(afp)
        #
        self.kwargs['dataset_info'] = self.get_dataset_info()
        num_classes = num_classes or len(self.kwargs['dataset_info']['categories'])
        self.num_classes = num_classes

    def download(self, path, split):
        return

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
        if self.kwargs['num_classes'] is not None:
            dataset_store.update(dict(color_map=self.get_color_map()))
        #
        return dataset_store

    def get_num_classes(self):
        return self.num_classes


class ModelMakerClassificationDataset(DatasetBase):
    def __init__(self, num_classes=None, download=False, num_frames=None, name='modelmaker',
                 annotation_prefix='instances', **kwargs):
        assert 'path' in kwargs and 'split' in kwargs, 'kwargs must have path and split'
        path = kwargs['path']
        split = kwargs['split']
        self.image_dir = kwargs.pop('image_dir', os.path.join(path, split))
        self.annotation_file = kwargs.pop('annotation_file', os.path.join(path, 'annotations', f'{annotation_prefix}_{split}.json'))
        with open(self.annotation_file) as afp:
            self.dataset_store = json.load(afp)
        #
        self.images_info = self.dataset_store['images']
        self.annotations_info = self.dataset_store['annotations']
        if num_classes is None:
            classes = self.dataset_store['categories']
            class_ids = [class_info['id'] for class_info in classes]
            class_ids_min = min(class_ids)
            num_classes = max(class_ids) - class_ids_min + 1
        #
        self.num_classes = num_classes
        self.annotations_info = self._find_annotations_info()
        max_frames = len(self.images_info)
        self.num_frames = min(num_frames, max_frames) if num_frames is not None else max_frames
        super().__init__(num_classes=num_classes, image_dir=self.image_dir, annotation_file=self.annotation_file,
                         download=False, num_frames=num_frames, name=name, **kwargs)
        self.kwargs['dataset_info'] = self.get_dataset_info()

    def download(self, path, split):
        return

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
        if self.kwargs['num_classes'] is not None:
            dataset_store.update(dict(color_map=self.get_color_map()))
        #
        return dataset_store

    def get_num_classes(self):
        return self.num_classes

    def __getitem__(self, idx, with_label=False, **kwargs):
        image_info = self.images_info[idx]
        filename = os.path.join(self.image_dir, image_info['file_name'])
        label = self.annotations_info[idx][0]['category_id']
        if with_label:
            return filename, label
        else:
            return filename

    def __len__(self):
        return min(self.num_frames, len(self.images_info)) if self.num_frames else len(self.images_info)

    def __call__(self, predictions, **kwargs):
        return self.evaluate(predictions, **kwargs)

    def evaluate(self, predictions, **kwargs):
        metric_tracker = utils.AverageMeter(name='accuracy_top1%')
        num_frames = min(self.num_frames, len(predictions))
        for n in range(num_frames):
            words = self.__getitem__(n, with_label=True)
            gt_label = int(words[1])
            accuracy = self.classification_accuracy(predictions[n], gt_label, **kwargs)
            metric_tracker.update(accuracy)
        #
        return {metric_tracker.name: metric_tracker.avg}

    def classification_accuracy(self, prediction, target, label_offset_pred=0, label_offset_gt=0,
                                multiplier=100.0, **kwargs):
        prediction = prediction + label_offset_pred
        target = target + label_offset_gt
        accuracy = 1.0 if (prediction == target) else 0.0
        accuracy = accuracy * multiplier
        return accuracy

    def _find_annotations_info(self):
        image_id_to_file_id_dict = dict()
        file_id_to_image_id_dict = dict()
        annotations_info_list = []
        for file_id, image_info in enumerate(self.dataset_store['images']):
            image_id = image_info['id']
            image_id_to_file_id_dict[image_id] = file_id
            file_id_to_image_id_dict[file_id] = image_id
            annotations_info_list.append([])
        #
        for annotation_info in self.dataset_store['annotations']:
            if annotation_info:
                image_id = annotation_info['image_id']
                file_id = image_id_to_file_id_dict[image_id]
                annotations_info_list[file_id].append(annotation_info)
            #
        #
        return annotations_info_list


class ModelMakerSegmentationDataset(DatasetBase):
    def __init__(self, num_classes=None, download=False, num_frames=None, with_background_class=True, name="modelmaker_seg", **kwargs):
        super().__init__(num_classes=num_classes, num_frames=num_frames, name=name, **kwargs)
        self.force_download = True if download == 'always' else False
        self.with_background_class = with_background_class
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

        dataset_folders = os.listdir(root)
        assert 'annotations' in dataset_folders, 'invalid path to coco dataset annotations'
        annotations_dir = os.path.join(root, 'annotations')

        shuffle = self.kwargs.get('shuffle', False)
        image_base_dir = 'images' if ('images' in dataset_folders) else ''
        image_base_dir = os.path.join(root, image_base_dir)
        self.image_dir = kwargs.pop('image_dir', image_base_dir)

        self.annotation_file = kwargs.pop('annotation_file', os.path.join(annotations_dir, f'{self.kwargs["annotation_prefix"]}_{split}.json'))

        with open(self.annotation_file) as afp:
            json_data = json.load(afp)

        self.coco_dataset = COCO(self.annotation_file)

        self.cat_ids = self.coco_dataset.getCatIds()
        self.img_ids = self.coco_dataset.getImgIds()

        self.min_class_id = min(self.cat_ids)
        if self.with_background_class and self.min_class_id > 0:
            self.num_classes = len(self.cat_ids) + 1
        else:
            self.num_classes = len(self.cat_ids)
        #
        self.categories = self.cat_ids

        img_ids_shortlisted = self._remove_images_without_annotations(self.img_ids)

        max_frames = len(img_ids_shortlisted)
        num_frames = self.kwargs.get('num_frames', None)
        num_frames = min(num_frames, max_frames) if num_frames is not None else max_frames

        imgs_list = list(self.coco_dataset.imgs.items())
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(imgs_list)
        #
        imgs_list_shortlisted = [(k, v) for k, v in imgs_list if k in img_ids_shortlisted]
        self.coco_dataset.imgs  = dict(imgs_list_shortlisted[:num_frames])

        max_frames = len(self.img_ids)
        num_frames = self.kwargs.get('num_frames', None)
        num_frames = min(num_frames, max_frames) if num_frames is not None else max_frames

        # self.cat_ids = self.coco_dataset.getCatIds()
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
        return

    def _get_root(self, path):
        path = path.rstrip('/')
        root = os.sep.join(os.path.split(path)[:-1])
        return root

    def get_num_classes(self):
        return self.num_classes

    def __getitem__(self, idx, with_label=False, label_as_array=False):
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
            self.encode_segmap(target)
            if not label_as_array:
                # write the label file to a temorary dir so that it can be used by evaluate()
                image_basename = os.path.basename(image_path)
                label_path = os.path.join(self.label_dir, image_basename)
                label_path = os.path.splitext(label_path)[0] + '.png'
                cv2.imwrite(label_path, target)
                return image_path, label_path
            else:
                return image_path, target
            #
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
        if not self.with_background_class and self.min_class_id > 0:
            # target[target == 0] = self.num_classes
            # target[target != 0] -= 1
            label_img[label_img == 0] = 255
            label_img[label_img != 255] -= 1
        #
        return label_img

    def __call__(self, predictions, **kwargs):
        return self.evaluate(predictions, **kwargs)

    def evaluate(self, predictions, **kwargs):
        cmatrix = None
        num_frames = min(self.num_frames, len(predictions))
        for n in range(num_frames):
            image_file, label_img = self.__getitem__(n, with_label=True, label_as_array=True)
            # label_img = PIL.Image.open(label_file)
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
        min_cat_id = min([cat['id'] for cat in dataset_store['categories']])
        if self.with_background_class and min_cat_id > 0:
            dataset_store['categories'] = [dict(id=0, supercategory=0, name='background')] + dataset_store['categories']
        #
        if self.kwargs['num_classes'] is not None:
            dataset_store.update(dict(color_map=self.get_color_map()))
        #
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
            obj["category_id"] = self.categories.index(obj["category_id"]) + 1
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
            # target[masks.sum(0) > 1] = 255
        else:
            target = np.zeros((h, w), dtype=np.uint8)
        #
        return image, target

    def _convert_poly_to_mask(self, segmentations, height, width):
        masks = []
        for polygons in segmentations:
            if len(polygons) != 1:
                polygons = [polygons]
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


from . import coco_kpt_det
class ModelMakerKeypointDetectionDataset(coco_kpt_det.COCOKeypointDetection):
    def __init__(self, num_classes=1, download=False, num_frames=None, name='modelmaker_kpt_det', num_keypoints=None,
                 annotation_prefix=None, **kwargs):
        assert 'path' in kwargs and 'split' in kwargs, 'kwargs must have path and split'
        path = kwargs['path']
        split = kwargs['split']
        image_dir = kwargs.pop('image_dir', None)
        annotation_file = kwargs.pop('annotation_file', os.path.join(path, 'annotations', f'{annotation_prefix}_{split}.json'))
        super().__init__(num_classes=num_classes, image_dir=None, annotation_file=annotation_file,
                         download=download, num_frames=num_frames, name=name, num_keypoints=num_keypoints, annotation_prefix=annotation_prefix, **kwargs)
        num_keypoints = self.num_keypoints
