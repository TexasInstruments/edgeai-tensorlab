#################################################################################
# Copyright (c) 2018-2021, Texas Instruments Incorporated - http://www.ti.com
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
#
#################################################################################

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
from torchvision.datasets import utils
from edgeai_torchmodelopt import xnn


__all__ = ['ADE20KSegmentation', 'ade20k_segmentation', 'ade20k_seg_noweights', 'ade20k_seg_class32']


###########################################
# config settings
def get_config():
    dataset_config = xnn.utils.ConfigNode()
    return dataset_config


###########################################
class ADE20KSegmentation:
    def __init__(self, dataset_config, num_classes=151, ignore_label=None, transforms=None, with_class_weights=True, 
                 download=False, additional_info=False, **kwargs):
        super().__init__()
        assert 'path' in kwargs and 'split' in kwargs, 'path and split must be provided'
        assert num_classes <= 151, 'maximum 151 classes (including background) are supported'
        #assert self.kwargs['split'] in ['training', 'validation']		
        self.transforms = transforms
        self.kwargs = kwargs

        path = kwargs['path']
        split = kwargs['split']
        if download:
            self.download(path, split)
        #

        self.kwargs['num_frames'] = self.kwargs.get('num_frames', None)
        self.name = "ADE20K"
        self.num_classes_ = num_classes
        self.ignore_label = ignore_label
        self.additional_info = additional_info
        self.label_dir_txt = os.path.join(self.kwargs['path'], 'objectInfo150.txt')
        self.load_classes()

        # if a color representation is needed
        self.color_map = xnn.utils.get_color_palette(num_classes)

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
        self.num_frames = min(self.kwargs['num_frames'], len(self.imgs)) \
            if (self.kwargs['num_frames'] is not None) else len(self.imgs)

        # class weights for training - not needed for inference
        if with_class_weights:
            with open(self.label_dir_txt) as fp:
                dataset_info = [line for line in fp]
                dataset_info = dataset_info[1:]
                dataset_info = [line.split() for line in dataset_info]
                dataset_freq = [float(line[1]) for line in dataset_info]
                # make space for background class and compute/put that freq
                dataset_freq.insert(0, 0.0)
                dataset_rem = 1.0 - sum(dataset_freq)
                if dataset_rem > 0.0:
                    dataset_freq[0] += dataset_rem
                #
                # all classes >= self.num_classes_ is mapped to 0
                dataset_freq[0] += sum(dataset_freq[self.num_classes_:])
                dataset_freq = dataset_freq[:self.num_classes_]
                dataset_freq = np.array(dataset_freq, dtype=np.float32)
                self.class_weights_ = np.mean(dataset_freq) / dataset_freq
            #
        else:
            self.class_weights_ = None
        #

    def class_weights(self):
        return self.class_weights_

    def download(self, path, split):
        root = path
        images_folder = os.path.join(path, 'images')
        annotations_folder = os.path.join(path, 'annotations')
        if os.path.exists(path) and os.path.exists(images_folder) and os.path.exists(annotations_folder):
            return
        #
        print('Important: Please visit the urls http://sceneparsing.csail.mit.edu/ '
              'https://groups.csail.mit.edu/vision/datasets/ADE20K/ and '
              'https://groups.csail.mit.edu/vision/datasets/ADE20K/terms/ '
              'to understand more about the ADE20K dataset '
              'and accept the terms and conditions under which it can be used. ')

        dataset_url = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
        root = root.rstrip('/')
        dataset_path = utils.download_and_extract_archive(dataset_url, download_root=root, extract_root=os.path.split(root)[0])
        return

    def __getitem__(self, idx):
        image_file = self.imgs[idx]
        label_file = self.labels[idx]
        image = PIL.Image.open(image_file).convert('RGB')
        label_img = PIL.Image.open(label_file)
        label_img = self.encode_segmap(label_img)
        # edgeailite transforms expect a list
        image = [image]
        label_img = [label_img]
        image_file = [image_file]
        label_file = [label_file]
        # apply the transform
        if self.transforms is not None:
            image, label_img = self.transforms(image, label_img)
        #
        if self.additional_info:
            return image, label_img, image_file, label_file
        else:
            return image, label_img

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

    def encode_segmap(self, label_img):
        label_img = label_img.convert('L')
        label_img = np.array(label_img)
        if self.ignore_label is not None:
            label_img[self.ignore_label] = 255
        #
        if self.num_classes_ < 151:
            label_img[label_img >= self.num_classes_] = 0
        #
        return label_img

    def evaluate(self, predictions, **kwargs):
        cmatrix = None
        for n in range(self.num_frames):
            image, label_img = self.__getitem__(n)
            # reshape prediction is needed
            output = predictions[n]
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


def ade20k_segmentation(dataset_config, path, split=None, num_classes=151, transforms=None, *args, **kwargs):
    dataset_config = get_config().merge_from(dataset_config)
    train_split = val_split = None
    split = ('training', 'validation') if split is None else split
    for split_name in split:
        if split_name == 'training':
            train_split = ADE20KSegmentation(dataset_config, path=path, split=split_name, num_classes=num_classes,
                                             transforms=transforms[0], *args, **kwargs)
        elif split_name == 'validation':
            val_split = ADE20KSegmentation(dataset_config, path=path, split=split_name, num_classes=num_classes,
                                           transforms=transforms[1], *args, **kwargs)
        else:
            pass
        #
    #
    return (train_split, val_split)


def ade20k_seg_noweights(*args, **kwargs):
    return ade20k_segmentation(*args, with_class_weights=False, **kwargs)


def ade20k_seg_class32(dataset_config, path, splits=None, num_classes=32, transforms=None, *args, **kwargs):
    return ade20k_segmentation(dataset_config, path, splits=splits, num_classes=num_classes,
                               transforms=transforms, *args, **kwargs)

