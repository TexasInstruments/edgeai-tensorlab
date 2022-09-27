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

"""
Reference:

M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele,
“The Cityscapes Dataset for Semantic Urban Scene Understanding,”
in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
https://www.cityscapes-dataset.com/
"""


import os
import glob
import random
import numpy as np
import PIL
from colorama import Fore
from .. import utils
from .dataset_base import *

__all__ = ['CityscapesSegmentation']

class CityscapesSegmentation(DatasetBase):
    def __init__(self, num_classes=19, download=False, num_frames=None, name="cityscapes", **kwargs):
        super().__init__(num_classes=num_classes, num_frames=num_frames, name=name, **kwargs)
        assert 'path' in self.kwargs and 'split' in self.kwargs, 'path and split must provided'
        path = self.kwargs['path']
        if not (os.path.exists(path) or os.path.isdir(path)) and download:
            print(f'{Fore.YELLOW}'
                  f'\nCityscapes dataset:'
                  f'\n    The Cityscapes Dataset for Semantic Urban Scene Understanding,'                  
                  f'\n        M. Cordts, et.al. in Proc. of the IEEE Conference on '
                  f'\n        Computer Vision and Pattern Recognition (CVPR), 2016.\n'
                  f'\n    Visit the following url to know more about the Cityscapes dataset: '
                  f'\n        https://www.cityscapes-dataset.com/  '
                  f'\n        and also to register and obtain the download links. '                  
                  f'{Fore.RESET}\n')
            assert False, f'input path {path} must contain the dataset'
        #

        # mapping for cityscapes 19 class segmentation
        self.num_classes = num_classes
        self.label_dict = {0:255, 1:255, 2:255, 3:255, 4:255, 5:255, 6:255, 7:0, 8:1, 9:255,
                     10:255, 11:2, 12:3, 13:4, 14:255, 15:255, 16:255, 17:5, 18:255, 19:6,
                     20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:255,
                     30:255, 31:16, 32:17, 33:18, 255:255}
        self.label_lut = self._create_lut()

        image_dir = os.path.join(self.kwargs['path'], 'leftImg8bit', self.kwargs['split'])
        images_pattern = os.path.join(image_dir, '*', '*.png')
        images = glob.glob(images_pattern)
        self.imgs = sorted(images)
        #
        label_dir = os.path.join(self.kwargs['path'], 'gtFine', self.kwargs['split'])
        labels_pattern = os.path.join(label_dir, '*', '*_labelIds.png')
        labels = glob.glob(labels_pattern)
        self.labels = sorted(labels)
        #
        assert len(self.imgs) == len(self.labels), 'length of images must be equal to the length of labels'

        shuffle = self.kwargs['shuffle'] if (isinstance(self.kwargs, dict) and 'shuffle' in self.kwargs) else False
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.imgs)
            random.seed(int(shuffle))
            random.shuffle(self.labels)
        #
        self.num_frames = self.kwargs['num_frames'] = min(self.kwargs['num_frames'], len(self.imgs)) \
            if (self.kwargs['num_frames'] is not None) else len(self.imgs)

    def __getitem__(self, idx, with_label=False):
        if with_label:
            image_file = self.imgs[idx]
            label_file = self.labels[idx]
            return image_file, label_file
        else:
            return self.imgs[idx]
        #

    def __len__(self):
        return self.num_frames

    def __call__(self, predictions, **kwargs):
        return self.evaluate(predictions, **kwargs)

    def evaluate(self, predictions, **kwargs):
        cmatrix = None
        num_frames = min(self.num_frames, len(predictions))
        for n in range(num_frames):
            image_file, label_file = self.__getitem__(n, with_label=True)
            # image = PIL.Image.open(image_file)
            label_img = PIL.Image.open(label_file)
            label_img = self.encode_segmap(label_img)

            output = predictions[n]
            output = output.astype(np.uint8)
            output = output[0] if (output.ndim > 2 and output.shape[0] == 1) else output
            output = output[:2] if (output.ndim > 2 and output.shape[2] == 1) else output

            cmatrix = utils.confusion_matrix(cmatrix, output, label_img, self.num_classes)
        #
        accuracy = utils.segmentation_accuracy(cmatrix)
        return accuracy

    def encode_segmap(self, label_img):
        if not isinstance(label_img, np.ndarray):
            # assumes it is PIL.Image
            label_img = label_img.convert('L')
            label_img = np.array(label_img)
        #
        label_img = self.label_lut[label_img]
        return label_img

    def _create_lut(self):
        if self.label_dict:
            lut = np.zeros(256, dtype=np.uint8)
            for k in range(256):
                lut[k] = k
            for k in self.label_dict.keys():
                lut[k] = self.label_dict[k]
            return lut
        else:
            return None
        #


if __name__ == '__main__':
    # from inside the folder jacinto_ai_benchmark, run the following:
    # python3 -m edgeai_benchmark.datasets.cityscapes
    # to create a converted dataset if you wish to load it using the dataset loader ImageSegmentation() in image_seg.py
    # to load it using CityscapesSegmentation dataset in this file, this conversion is not required.
    import shutil
    output_folder = './dependencies/datasets/cityscapes-converted'
    path = './dependencies/datasets/cityscapes'
    split = 'val'
    cityscapes_seg = CityscapesSegmentation(path=path, split=split)
    num_frames = len(cityscapes_seg)

    images_output_folder = output_folder
    labels_output_folder = output_folder
    os.makedirs(images_output_folder, exist_ok=True)
    os.makedirs(labels_output_folder, exist_ok=True)

    output_filelist = os.path.join(output_folder, f'{split}.txt')
    with open(output_filelist, 'w') as list_fp:
        for n in range(num_frames):
            image_path, label_path = cityscapes_seg.__getitem__(n, with_label=True)
            label_img = PIL.Image.open(label_path)
            label_img = cityscapes_seg.encode_segmap(label_img)

            image_output_filename = image_path.replace(path, images_output_folder)
            assert image_output_filename != image_path, f'output iamge path is incorrect {image_output_filename}'
            label_output_filename = label_path.replace(path, labels_output_folder)
            assert label_output_filename != label_path, f'output label path is incorrect {label_output_filename}'

            os.makedirs(os.path.dirname(image_output_filename), exist_ok=True)
            os.makedirs(os.path.dirname(label_output_filename), exist_ok=True)

            shutil.copy2(image_path, image_output_filename)
            label_img = PIL.Image.fromarray(label_img)
            label_img.save(label_output_filename)

            list_fp.write(f'{image_output_filename} {label_output_filename}\n')
        #
    #
