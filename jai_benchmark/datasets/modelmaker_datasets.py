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

from .dataset_base import *
from . import coco_det


class ModelMakerDetectionDataset(coco_det.COCODetection):
    def __init__(self, num_classes=90, download=False, num_frames=None, name='modelmaker', **kwargs):
        assert 'path' in kwargs and 'split' in kwargs, 'kwargs must have path and split'
        path = kwargs['path']
        split = kwargs['split']
        image_dir = os.path.join(path, split)
        annotation_file = os.path.join(path, 'annotations', f'instances_{split}.json')
        super().__init__(num_classes=num_classes, image_dir=image_dir, annotation_file=annotation_file,
                         download=False, num_frames=num_frames, name=name, **kwargs)

    def download(self, path, split):
        return


class ModelMakerClassificationDataset(DatasetBase):
    def __init__(self, num_classes=None, download=False, num_frames=None, name='modelmaker', **kwargs):
        assert 'path' in kwargs and 'split' in kwargs, 'kwargs must have path and split'
        path = kwargs['path']
        split = kwargs['split']
        self.image_dir = os.path.join(path, split)
        self.annotation_file = os.path.join(path, 'annotations', f'labels_{split}.json')
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
        if num_frames is None:
            num_frames = len(self.images_info)
        #
        self.num_frames = num_frames
        super().__init__(num_classes=num_classes, image_dir=self.image_dir, annotation_file=self.annotation_file,
                         download=False, num_frames=num_frames, name=name, **kwargs)

    def download(self, path, split):
        return

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
        return {metric_tracker.name:metric_tracker.avg}

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