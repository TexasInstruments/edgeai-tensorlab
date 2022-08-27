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

import os
import random
from colorama import Fore
from .. import utils
from .dataset_base import *


class ImageClassification(DatasetBase):
    def __init__(self, download=False, dest_dir=None, num_frames=None, name=None, **kwargs):
        super().__init__(num_frames=num_frames, name=name, **kwargs)
        self.force_download = True if download == 'always' else False
        assert 'path' in self.kwargs and 'split' in self.kwargs, 'path and split must be provided in kwargs'
        assert 'num_classes' in self.kwargs, f'num_classes must be provided while creating {self.__class__.__name__}'
        assert name is not None, 'Please provide a name for this dataset'

        path = self.kwargs['path']
        split_file = self.kwargs['split']

        # download the data if needed
        if download:
            if (not self.force_download) and os.path.exists(path):
                print(utils.log_color('\nINFO', 'dataset exists - will reuse', path))
            else:
                self.download(path, split_file)
            #
            if hasattr(self, 'download_extra'):
                self.download_extra(path, split_file)
            #
        #
        assert os.path.exists(path) and os.path.isdir(path), \
            utils.log_color('\nERROR', 'dataset path is empty', path)

        # create list of images and classes
        self.imgs = utils.get_data_list(input=kwargs, dest_dir=dest_dir)
        self.num_frames = self.kwargs['num_frames'] = self.kwargs.get('num_frames',len(self.imgs))
        shuffle = self.kwargs.get('shuffle', False)
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.imgs)
        #

    def download(self, path, split_file):
        return None

    def __getitem__(self, idx, **kwargs):
        with_label = kwargs.get('with_label', False)
        words = self.imgs[idx].split(' ')
        image_name = words[0]
        if with_label:
            assert len(words)>0, f'ground truth requested, but missing at the dataset entry for {words}'
            label = int(words[1])
            return image_name, label
        else:
            return image_name
        #

    def __len__(self):
        return self.num_frames

    def __call__(self, predictions, **kwargs):
        return self.evaluate(predictions, **kwargs)

    def evaluate(self, predictions, **kwargs):
        in_lines = self.imgs
        metric_tracker = utils.AverageMeter(name='accuracy_top1%')
        num_frames = min(self.num_frames, len(predictions))
        for n in range(num_frames):
            words = in_lines[n].split(' ')
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

