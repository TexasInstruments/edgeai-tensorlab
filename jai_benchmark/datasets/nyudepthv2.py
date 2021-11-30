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
import glob
import random
import numpy as np
import PIL
from colorama import Fore
from .. import utils
from .dataset_base import *

class NYUDepthV2(DatasetBase):
    def __init__(self, num_classes=151, ignore_label=None, download=False, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)

        self.kwargs['num_frames'] = self.kwargs.get('num_frames', None)
        self.name = "NYUDEPTHV2"
        self.ignore_label = ignore_label
        #self.label_dir_txt = os.path.join(self.kwargs['path'], 'objectInfo150.txt')

        image_dir = os.path.join(self.kwargs['path'], self.kwargs['split'], 'images')
        images_pattern = os.path.join(image_dir, '*.jpg')
        images = glob.glob(images_pattern)
        self.imgs = sorted(images)

        self.num_frames = min(self.kwargs['num_frames'], len(self.imgs)) \
            if (self.kwargs['num_frames'] is not None) else len(self.imgs)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx, with_label=False):
        if with_label:
            image_file = self.imgs[idx]
            label_file = self.labels[idx]
            return image_file, label_file
        else:
            return self.imgs[idx]
        #

    def evaluate(self, predictions, threshold, **kwargs):
        bad_pixels = 0.0
        num_frames = min(self.num_frames, len(predictions))
        for n in range(num_frames):
            image_file, label_file = self.__getitem__(n, with_label=True)
            label_img = PIL.Image.open(label_file)
            mask = np.min(label_img, predictions[n]) != 0 

            delta = np.min(
                predictions[n][mask] / label_img[mask], 
                label_img[mask] / predictions[n][mask]
            )
            bad_pixels_in_img = delta > threshold
            bad_pixels += bad_pixels_in_img.sum() / mask.sum()
        #

        bad_pixels /= n
        return bad_pixels