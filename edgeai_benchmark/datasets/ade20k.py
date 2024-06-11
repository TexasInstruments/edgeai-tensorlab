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

"""
Reference:

Scene Parsing through ADE20K Dataset.
Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso and Antonio Torralba.
Computer Vision and Pattern Recognition (CVPR), 2017. Semantic Understanding of Scenes through ADE20K Dataset.

Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso and Antonio Torralba.
International Journal on Computer Vision (IJCV).

https://groups.csail.mit.edu/vision/datasets/ADE20K/, http://sceneparsing.csail.mit.edu/
"""


import os
import glob
import random
import numpy as np
import PIL
from colorama import Fore
from .. import utils
from .dataset_base import *

__all__ = ['ADE20KSegmentation']

class ADE20KSegmentation(DatasetBase):
    def __init__(self, num_classes=151, ignore_label=None, download=False, num_frames=None, name="ADE20K", **kwargs):
        super().__init__(num_classes=num_classes, num_frames=num_frames, name=name, **kwargs)
        self.force_download = True if download == 'always' else False
        assert 'path' in self.kwargs and 'split' in self.kwargs, 'path and split must be provided'
        assert num_classes <= 151, 'maximum 151 classes (including background) are supported'

        path = self.kwargs['path']
        split = kwargs['split']
        if download:
            self.download(path, split)
        #

        self.num_classes_ = num_classes
        self.ignore_label = ignore_label
        self.label_dir_txt = os.path.join(self.kwargs['path'], 'objectInfo150.txt')
        self.load_classes()

        # if a color representation is needed
        self.color_map = utils.get_color_palette(num_classes)

        image_dir = os.path.join(self.kwargs['path'], 'images', self.kwargs['split'])
        images_pattern = os.path.join(image_dir, '*.jpg')
        images = glob.glob(images_pattern)
        self.imgs = sorted(images)
        #
        label_dir = os.path.join(self.kwargs['path'], 'annotations', self.kwargs['split'])
        labels_pattern = os.path.join(label_dir, '*.png')
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

        self.dataset_store =dict(
            info=dict(url='https://groups.csail.mit.edu/vision/datasets/ADE20K/, https://github.com/CSAILVision/ADE20K',
                                           description='Scene parsing data and part segmentation data derived from ADE20K dataset',
                      contributor='MIT CSAIL / MIT Scene Parsing Benchmark, ADE20K is composed of more than 27K images from the SUN and Places databases',
                      year='2016'),
                                 categories=[dict(id=class_entry_value, name=class_entry_key)  for class_entry_key, class_entry_value in self.classes.items()])
        self.kwargs['dataset_info'] = self.get_dataset_info()

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

    def download(self, path, split):
        root = path
        images_folder = os.path.join(path, 'images')
        annotations_folder = os.path.join(path, 'annotations')
        if (not self.force_download) and os.path.exists(path) and os.path.exists(images_folder) and os.path.exists(annotations_folder):
            print(utils.log_color('\nINFO', 'dataset exists - will reuse', path))
            return
        #
        print(utils.log_color('\nINFO', 'downloading and preparing dataset', path + ' This may take some time.'))
        print(f'{Fore.YELLOW}'
              f'\nADE20K Dataset:'
              f'\n    Scene Parsing through ADE20K Dataset.'
              f'\n       Bolei Zhou, et.al., Computer Vision and Pattern Recognition (CVPR), 2017. '
              f'\n    Semantic Understanding of Scenes through ADE20K Dataset.'
              f'\n        Bolei Zhou, et.al, International Journal on Computer Vision (IJCV).\n'
              f'\n    Visit the following urls to know more about ADE20K dataset: '            
              f'\n        http://sceneparsing.csail.mit.edu/ '
              f'\n        https://groups.csail.mit.edu/vision/datasets/ADE20K/ '
              f'{Fore.RESET}\n')

        dataset_url = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
        root = root.rstrip('/')
        download_root = os.path.join(root, 'download')
        dataset_path = utils.download_file(dataset_url, root=download_root, extract_root=os.path.split(root)[0],
                                           force_download=self.force_download)
        print(utils.log_color('\nINFO', 'dataset ready', path))
        return

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

    def num_classes(self):
        return [self.num_classes_]

    def decode_segmap(self, seg_img):
        r = seg_img.copy()
        g = seg_img.copy()
        b = seg_img.copy()
        for l in range(0, self.num_classes_):
            r[seg_img == l] = self.color_map[l][0]
            g[seg_img == l] = self.color_map[l][1]
            b[seg_img == l] = self.color_map[l][2]
        #

        rgb = np.zeros((seg_img.shape[0], seg_img.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, label_img, label_offset_target=0):
        label_img = label_img.convert('L')
        label_img = np.array(label_img)
        label_img = label_img + label_offset_target
        if self.ignore_label is not None:
            label_img[self.ignore_label] = 255
        #
        if self.num_classes_ < 151:
            label_img[label_img >= self.num_classes_] = 0
        #
        return label_img

    def evaluate(self, predictions, **kwargs):
        label_offset_target = kwargs.get('label_offset_target', 0)
        label_offset_pred = kwargs.get('label_offset_pred', 0)
        cmatrix = None
        num_frames = min(self.num_frames, len(predictions))
        for n in range(num_frames):
            image_file, label_file = self.__getitem__(n, with_label=True)
            label_img = PIL.Image.open(label_file)
            label_img = self.encode_segmap(label_img, label_offset_target=label_offset_target)
            # reshape prediction is needed
            output = predictions[n]+label_offset_pred
            output = output.astype(np.uint8)
            output = output[0] if (output.ndim > 2 and output.shape[0] == 1) else output
            output = output[:2] if (output.ndim > 2 and output.shape[2] == 1) else output
            # compute metric
            cmatrix = utils.confusion_matrix(cmatrix, output, label_img, self.num_classes_)
        #
        accuracy = utils.segmentation_accuracy(cmatrix)
        return accuracy

    def load_classes(self):
        #ade20k_150_classes_url = "https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/objectInfo150.csv"
        with open(self.label_dir_txt) as f:
            list_ade20k_classes = list(map(lambda x: x.split("\t")[-1], f.read().split("\n")))[1:self.num_classes_+1]
            self.classes_reverse = dict(zip([i for i in range(1,self.num_classes_+1)],list_ade20k_classes))
            self.classes = dict(zip(list_ade20k_classes,[i for i in range(1,self.num_classes_+1)]))
        #

