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


import numbers
import os
import random
import json
import shutil
import tempfile
import numpy as np
from colorama import Fore
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..config_utils.dataset_utils import *
from .. import utils
from .dataset_base import *

__all__ = ['COCODetection', 'coco_det_label_offset_80to90', 'coco_det_label_offset_90to90']


class COCODetection(DatasetBase):
    def __init__(self, num_classes=90, download=False, image_dir=None, annotation_file=None, num_frames=None, name='coco', **kwargs):
        super().__init__(num_classes=num_classes, num_frames=num_frames, name=name, **kwargs)
        if image_dir is None or annotation_file is None:
            self.force_download = True if download == 'always' else False
            assert 'path' in self.kwargs and 'split' in self.kwargs, 'kwargs must have path and split'
            path = self.kwargs['path']
            split = self.kwargs['split']
            if download:
                self.download(path, split)
            #

            dataset_folders = os.listdir(self.kwargs['path'])
            assert 'annotations' in dataset_folders, 'invalid path to coco dataset annotations'
            annotations_dir = os.path.join(self.kwargs['path'], 'annotations')

            image_base_dir = 'images' if ('images' in dataset_folders) else ''
            image_base_dir = os.path.join(self.kwargs['path'], image_base_dir)
            image_split_dirs = os.listdir(image_base_dir)
            assert self.kwargs['split'] in image_split_dirs, f'invalid path to coco dataset images/split {kwargs["split"]}'
            self.image_dir = os.path.join(image_base_dir, self.kwargs['split'])
            self.annotation_file = os.path.join(annotations_dir, f'instances_{self.kwargs["split"]}.json')
        else:
            self.image_dir = image_dir
            self.annotation_file = annotation_file
        #
        self._load_dataset()
        with open(self.annotation_file) as afp:
            self.dataset_store = json.load(afp)
        #
        self.kwargs['dataset_info'] = self.get_dataset_info()

    def _load_dataset(self):
        shuffle = self.kwargs.get('shuffle', False)
        self.coco_dataset = COCO(self.annotation_file)
        filter_imgs = self.kwargs['filter_imgs'] if 'filter_imgs' in self.kwargs else None
        if isinstance(filter_imgs, str):
            # filter images with the given list
            filter_imgs = os.path.join(self.kwargs['path'], filter_imgs)
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

        max_frames = len(self.coco_dataset.imgs)
        num_frames = self.kwargs.get('num_frames', None)
        num_frames = min(num_frames, max_frames) if num_frames is not None else max_frames

        imgs_list = list(self.coco_dataset.imgs.items())
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(imgs_list)
        #
        self.coco_dataset.imgs = {k:v for k,v in imgs_list[:num_frames]}

        self.cat_ids = self.coco_dataset.getCatIds()
        self.img_ids = self.coco_dataset.getImgIds()
        self.num_frames = self.kwargs['num_frames'] = num_frames
        self.tempfiles = []

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
            return image_path, None
        else:
            return image_path

    def __len__(self):
        return min(self.num_frames, len(self.img_ids)) if self.num_frames else len(self.img_ids)

    def __del__(self):
        for t in self.tempfiles:
            t.cleanup()
        #

    def __call__(self, predictions, **kwargs):
        return self.evaluate(predictions, **kwargs)

    def evaluate(self, predictions, **kwargs):
        label_offset = kwargs.get('label_offset_pred', 0)
        #run_dir = kwargs.get('run_dir', None)
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir = temp_dir_obj.name
        self.tempfiles.append(temp_dir_obj)

        #os.makedirs(run_dir, exist_ok=True)
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
        coco_ap = 0.0
        coco_ap50 = 0.0
        if len(detections_formatted_list) > 0:
            detection_file = os.path.join(temp_dir, 'detection_results.json')
            with open(detection_file, 'w') as det_fp:
                json.dump(detections_formatted_list, det_fp)
            #
            cocoDet = self.coco_dataset.loadRes(detection_file)
            cocoEval = COCOeval(self.coco_dataset, cocoDet, iouType='bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            coco_ap = cocoEval.stats[0]
            coco_ap50 = cocoEval.stats[1]
        #
        accuracy = {'accuracy_ap[.5:.95]%': coco_ap*100.0, 'accuracy_ap50%': coco_ap50*100.0}
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

    def _format_detections(self, bbox_label_score, image_id, label_offset=0, class_map=None):
        if class_map is not None:
            assert bbox_label_score[4] in class_map, 'invalid prediction label or class_map'
            bbox_label_score[4] = class_map[bbox_label_score[4]]
        #
        bbox_label_score[4] = self._detection_label_to_catid(bbox_label_score[4], label_offset)
        output_dict = dict()
        image_id = self.img_ids[image_id]
        output_dict['image_id'] = image_id
        det_bbox = bbox_label_score[:4]      # json is not support for ndarray - convert to list
        det_bbox = self._xyxy2xywh(det_bbox) # can also be done in postprocess pipeline
        det_bbox = self._to_list(det_bbox)
        output_dict['bbox'] = det_bbox
        output_dict['category_id'] = int(bbox_label_score[4])
        output_dict['score'] = float(bbox_label_score[5])
        return output_dict

    def _detection_label_to_catid(self, label, label_offset):
        if isinstance(label_offset, (list,tuple)):
            label = int(label)
            assert label<len(label_offset), 'label_offset is a list/tuple, but its size is smaller than the detected label'
            label = label_offset[label]
        elif isinstance(label_offset, dict):
            if np.isnan(label) or int(label) not in label_offset.keys():
                #print(utils.log_color('\nWARNING', 'detection incorrect', f'detected label: {label}'
                #                                                          f' is not in label_offset dict'))
                label = 0
            else:
                label = label_offset[int(label)]
            #
        elif isinstance(label_offset, numbers.Number):
            label = int(label + label_offset)
        else:
            label = int(label)
            assert label<len(self.cat_ids), \
                'the detected label could not be mapped to the 90 COCO categories using the default COCO.getCatIds()'
            label = self.cat_ids[label]
        #
        return label

    def _to_list(self, bbox):
        bbox = [float(x) for x in bbox]
        return bbox

    def _xyxy2xywh(self, bbox):
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        return bbox



################################################################################################
if __name__ == '__main__':
    # from inside the folder jacinto_ai_benchmark, run the following:
    # python -m edgeai_benchmark.datasets.coco_det
    # to create a converted dataset if you wish to load it using the dataset loader ImageDetection() in image_det.py
    # to load it using CocoSegmentation dataset in this file, this conversion is not required.
    import shutil
    output_folder = './dependencies/datasets/coco-det-converted'
    split = 'val2017'
    coco_seg = COCODetection(path='./dependencies/datasets/coco', split=split)
    num_frames = len(coco_seg)

    images_output_folder = os.path.join(output_folder, split, 'images')
    labels_output_folder = os.path.join(output_folder, split, 'labels')
    os.makedirs(images_output_folder)
    os.makedirs(labels_output_folder)

    output_filelist = os.path.join(output_folder, f'{split}.txt')
    with open(output_filelist, 'w') as list_fp:
        for n in range(num_frames):
            image_path, label_path = coco_seg.__getitem__(n, with_label=True)
            # TODO: labels are not currently written to list file
            image_output_filename = os.path.join(images_output_folder, os.path.basename(image_path))
            shutil.copy2(image_path, image_output_filename)
            list_fp.write(f'images/{os.path.basename(image_output_filename)}\n')
        #
    #
