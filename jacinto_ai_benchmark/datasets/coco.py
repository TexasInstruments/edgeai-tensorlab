import os
import random
import json
import tempfile
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .. import utils

__all__ = ['COCODetection', 'COCOSegmentation']

class COCODetection():
    def __init__(self, inData, num_imgs=None):
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

        filter_imgs = inData['filter_imgs'] if 'filter_imgs' in inData else None
        if isinstance(filter_imgs, str):
            # filter images with the given list
            filter_imgs = os.path.join(inData['path'], filter_imgs)
            with open(filter_imgs) as filter_fp:
                filter = [int(id) for id in list(filter_fp)]
                orig_keys = list(self.coco_dataset.imgs)
                orig_keys = [k for k in orig_keys if k in filter]
                self.coco_dataset.imgs = {k: self.coco_dataset.imgs[k] for k in orig_keys}
            #
        elif filter_imgs:
            # filter and use images with gt only
            sel_keys = []
            for img_key, img_anns in self.coco_dataset.imgToAnns.items():
                if len(img_anns) > 0:
                    sel_keys.append(img_key)
                #
            #
            self.coco_dataset.imgs = {k: self.coco_dataset.imgs[k] for k in sel_keys}
        #

        if num_imgs is not None:
            orig_keys = list(self.coco_dataset.imgs)[:num_imgs]
            self.coco_dataset.imgs = {k:self.coco_dataset.imgs[k] for k in orig_keys}
        #
        self.cat_ids = self.coco_dataset.getCatIds()
        self.img_ids = self.coco_dataset.getImgIds()
        imgs = []
        for img_id in self.img_ids:
            img = self.coco_dataset.loadImgs([img_id])[0]
            imgs.append(os.path.join(image_dir, img['file_name']))
        #
        self.imgs = imgs
        self.num_imgs = min(num_imgs, len(self.imgs)) if (num_imgs is not None) else len(self.imgs)
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.imgs)
        #

    def __getitem__(self, idx):
        return self.imgs[idx]

    def __len__(self):
        return self.num_imgs

    def __call__(self, predictions, label_offset=0, result_dir=None):
        return self.evaluate(predictions, label_offset, result_dir)

    def get_imgs(self):
        return self.imgs

    def evaluate(self, predictions, label_offset=0, result_dir=None):
        result_dir = tempfile.TemporaryDirectory().name if result_dir is None else result_dir
        detections_formatted_list = []
        for frame_idx, det_frame in enumerate(predictions):
            for det_id, det in enumerate(det_frame):
                det = self._format_detections(det, frame_idx, label_offset=label_offset)
                category_id = det['category_id'] if isinstance(det, dict) else det[4]
                if category_id >= 1: # final coco categories start from 1
                    detections_formatted_list.append(det)
                #
            #
        #
        if len(detections_formatted_list) > 0:
            detection_file = os.path.join(result_dir, 'detection_results.json')
            with open(detection_file, 'w') as det_fp:
                json.dump(detections_formatted_list, det_fp)
            #
            cocoDet = self.coco_dataset.loadRes(detection_file)
            cocoEval = COCOeval(self.coco_dataset, cocoDet, iouType='bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            coco_ap = cocoEval.stats[0]
        else:
            coco_ap = 0.0
        #
        return coco_ap

    def _format_detections(self, det_bbox_label_score, idx, format='json_serializable', label_offset=0, class_map=None):
        if class_map is not None:
            if det_bbox_label_score[4] in class_map.keys():
                det_bbox_label_score[4] = class_map[det_bbox_label_score[4]]
            else:
                print("Prediction of other classes")
            #
        #
        det_bbox_label_score[4] = self._detection_label_to_catid(det_bbox_label_score[4], label_offset)
        if format == 'json_serializable':
            output_dict = dict()
            image_id = self.img_ids[idx]
            output_dict['image_id'] = image_id
            output_dict['bbox'] = self._xyxy2xywh(det_bbox_label_score[:4])
            output_dict['category_id'] = int(det_bbox_label_score[4])
            output_dict['score'] = float(det_bbox_label_score[5])
            return output_dict
        else:
            return det_bbox_label_score

    def _detection_label_to_catid(self, label, label_offset):
        if isinstance(label_offset, (list,tuple)):
            label = int(label)
            assert label<len(label_offset), 'label_offset is a list/tuple, but its size is smaller than the detected label'
            label = label_offset[label]
        elif isinstance(label_offset, dict):
            label = int(label)
            assert label in label_offset.keys(), 'label_offset is a dict, but the detected label was not one of its keys'
            label = label_offset[label]
        else:
            label = int(label - label_offset)
            assert label<len(self.cat_ids), \
                'the detected label could not be mapped to the 90 COCO categories using the default COCO.getCatIds()'
            label = self.cat_ids[label]
        #
        return label

    def _xyxy2xywh(self, bbox):
        bbox = [float(x) for x in bbox]
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        return bbox


'''
Modified from: https://github.com/pytorch/vision
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016, 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
from pycocotools import mask as coco_mask
import copy
import numpy as np
import PIL
import cv2
class COCOSegmentation():
    def __init__(self, inData, num_imgs=None, num_classes=21):
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

    def evaluate(self, outputs):
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

