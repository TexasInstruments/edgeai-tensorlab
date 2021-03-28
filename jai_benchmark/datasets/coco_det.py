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


import numbers
import os
import random
import json
import shutil
import tempfile
from colorama import Fore
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .. import utils

__all__ = ['COCODetection']


class COCODetection(utils.ParamsBase):
    def __init__(self, download=False, **kwargs):
        super().__init__()
        self.force_download = True if download == 'always' else False
        self.kwargs = kwargs
        assert 'path' in kwargs and 'split' in kwargs, 'kwargs must have path and split'
        path = kwargs['path']
        split = kwargs['split']
        if download:
            self.download(path, split)
        #

        dataset_folders = os.listdir(kwargs['path'])
        assert 'annotations' in dataset_folders, 'invalid path to coco dataset annotations'
        annotations_dir = os.path.join(kwargs['path'], 'annotations')

        shuffle = kwargs.get('shuffle', False)
        image_base_dir = 'images' if ('images' in dataset_folders) else ''
        image_base_dir = os.path.join(kwargs['path'], image_base_dir)
        image_split_dirs = os.listdir(image_base_dir)
        assert kwargs['split'] in image_split_dirs, f'invalid path to coco dataset images/split {kwargs["split"]}'
        self.image_dir = os.path.join(image_base_dir, kwargs['split'])

        self.coco_dataset = COCO(os.path.join(annotations_dir, f'instances_{kwargs["split"]}.json'))

        filter_imgs = kwargs['filter_imgs'] if 'filter_imgs' in kwargs else None
        if isinstance(filter_imgs, str):
            # filter images with the given list
            filter_imgs = os.path.join(kwargs['path'], filter_imgs)
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
        num_frames = kwargs.get('num_frames', None)
        num_frames = min(num_frames, max_frames) if num_frames is not None else max_frames

        imgs_list = list(self.coco_dataset.imgs.items())
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(imgs_list)
        #
        self.coco_dataset.imgs = {k:v for k,v in imgs_list[:num_frames]}

        self.cat_ids = self.coco_dataset.getCatIds()
        self.img_ids = self.coco_dataset.getImgIds()
        self.num_frames = num_frames
        self.tempfiles = []
        # call the utils.ParamsBase.initialize()
        super().initialize()

    def download(self, path, split):
        root = path
        images_folder = os.path.join(path, split)
        annotations_folder = os.path.join(path, 'annotations')
        if (not self.force_download) and os.path.exists(path) and \
                os.path.exists(images_folder) and os.path.exists(annotations_folder):
            print(f'{Fore.CYAN}INFO:{Fore.YELLOW} dataset exists - will reuse:{Fore.RESET} {path}')
            return
        #
        print(f'{Fore.YELLOW}'
              f'\nCOCO Dataset:'
              f'\n    Microsoft COCO: Common Objects in Context, '
              f'\n        Tsung-Yi Lin, et.al. https://arxiv.org/abs/1405.0312\n'
              f'\nPlease visit the url: https://cocodataset.org/ and '
              f'\n    to know more about the COCO dataset and understand '
              f'\n    the terms and conditions under which it can be used.'
              f'{Fore.RESET}\n')

        dataset_url = 'http://images.cocodataset.org/zips/val2017.zip'
        extra_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
        download_root = os.path.join(root, 'download')
        dataset_path = utils.download_file(dataset_url, root=download_root, extract_root=root)
        extra_path = utils.download_file(extra_url, root=download_root, extract_root=root)
        return

    def _get_root(self, path):
        path = path.rstrip('/')
        root = os.sep.join(os.path.split(path)[:-1])
        return root

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = self.coco_dataset.loadImgs([img_id])[0]
        image_path = os.path.join(self.image_dir, img['file_name'])
        return image_path

    def __len__(self):
        return self.num_frames

    def __del__(self):
        for t in self.tempfiles:
            t.cleanup()
        #

    def __call__(self, predictions, **kwargs):
        return self.evaluate(predictions, **kwargs)

    def evaluate(self, predictions, **kwargs):
        label_offset = kwargs.get('label_offset_pred', 0)
        run_dir = kwargs.get('run_dir', None)
        if run_dir is None:
            temp_dir = tempfile.TemporaryDirectory()
            run_dir = temp_dir.name
            self.tempfiles.append(temp_dir)
        #
        os.makedirs(run_dir, exist_ok=True)
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
            detection_file = os.path.join(run_dir, 'detection_results.json')
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
            label = int(label)
            assert label in label_offset.keys(), f'label_offset is a dict, but the detected label {label} was not one of its keys'
            label = label_offset[label]
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

