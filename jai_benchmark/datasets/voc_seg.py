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

The PASCAL Visual Object Classes (VOC) Challenge, Everingham,
M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman, A.,
International Journal of Computer Vision, 88(2), 303-338, 2010,
http://host.robots.ox.ac.uk/pascal/VOC/
"""


import os
import glob
import random
import numpy as np
import PIL
import cv2
from colorama import Fore
from .. import utils
from .dataset_base import *

__all__ = ['VOC2012Segmentation']

class VOC2012Segmentation(DatasetBase):
    def __init__(self, num_classes=21, ignore_label=255, download=False, num_frames=None, name='voc2012', **kwargs):
        super().__init__(num_classes=num_classes, num_frames=num_frames, name=name, **kwargs)
        self.force_download = True if download == 'always' else False
        assert 'path' in self.kwargs and 'split' in self.kwargs, 'kwargs must have path and split'

        path = self.kwargs['path']
        split = self.kwargs['split']
        if download:
            self.download(path, split)
        #

        self.kwargs['num_frames'] = self.kwargs.get('num_frames', None)

        # mapping for voc 21 class segmentation
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        list_txt = os.path.join(self.kwargs['path'], 'ImageSets', 'Segmentation', self.kwargs['split']+'.txt')
        file_indexes = list(open(list_txt))

        base_path_images = os.path.join(self.kwargs['path'], 'JPEGImages')
        images = [base_path_images + '/' +file_index.rstrip() + '.jpg' for file_index in file_indexes]
        self.imgs = sorted(images)
        #
        base_path_labels = os.path.join(self.kwargs['path'], 'SegmentationClassRaw')
        labels = [base_path_labels + '/' +file_index.rstrip() + '.png' for file_index in file_indexes]
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

    def download(self, path, split=None):
        root = path
        extract_root = os.sep.join(root.split(os.sep)[:-2])
        download_root = os.path.join(os.sep.join(root.split(os.sep)[:-1]), 'download')
        images_folder = os.path.join(path, 'JPEGImages')
        imagesets_folder = os.path.join(path, 'ImageSets')
        segmentations_folder = os.path.join(path, 'SegmentationClassRaw')
        annotations_folder = os.path.join(path, 'Annotations') # not really required for segmentation
        if (not self.force_download) and os.path.exists(path) and \
                os.path.exists(imagesets_folder) and os.path.exists(images_folder) \
                and os.path.exists(segmentations_folder) and os.path.exists(annotations_folder):
            print(utils.log_color('\nINFO', 'dataset exists - will reuse', path))
            return
        #
        print(utils.log_color('\nINFO', 'downloading and preparing dataset', path + ' This may take some time.'))
        print(f'{Fore.YELLOW}'
              f'\nPascal VOC 2012 Dataset (VOC2012): '
              f'\n    The PASCAL Visual Object Classes (VOC) Challenge, '
              f'\n        Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman, A.,'
              f'\n        International Journal of Computer Vision, 88(2), 303-338, 2010,\n'
              f'\n    Visit the following url to know more about the Pascal VOC dataset. '
              f'\n        http://host.robots.ox.ac.uk/pascal/VOC/'
              f'{Fore.RESET}\n')

        dataset_url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar'
        extra_url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
        dataset_path = utils.download_file(dataset_url, root=download_root, extract_root=extract_root)
        extra_path = utils.download_file(extra_url, root=download_root, extract_root=extract_root)
        self.convert_dataset(root)
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

    def evaluate(self, predictions, **kwargs):
        cmatrix = None
        num_frames = min(self.num_frames, len(predictions))
        for n in range(num_frames):
            image_file, label_file = self.__getitem__(n, with_label=True)
            # image = PIL.Image.open(image_file)
            label_img = PIL.Image.open(label_file)
            label_img = label_img.convert('L')
            label_img = np.array(label_img)
            #label_img = self.label_lut[label_img]

            output = predictions[n]
            output = output.astype(np.uint8)
            output = output[0] if (output.ndim > 2 and output.shape[0] == 1) else output
            output = output[:2] if (output.ndim > 2 and output.shape[2] == 1) else output

            cmatrix = utils.confusion_matrix(cmatrix, output, label_img, self.num_classes)
        #
        accuracy = utils.segmentation_accuracy(cmatrix)
        return accuracy

    ############################################################
    # converts the PASCALVOC segmentation groundtruth from color format to raw format.
    # Source: https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/remove_gt_colormap.py
    ###########################################################
    def _remove_colormap(self, filename):
      """Removes the color map from the annotation.
      Args:
        filename: Ground truth annotation filename.
      Returns:
        Annotation without color map.
      """
      return np.array(PIL.Image.open(filename))

    def _save_annotation(self, annotation, filename):
      """Saves the annotation as png file.
      Args:
        annotation: Segmentation annotation.
        filename: Output filename.
      """
      pil_image = PIL.Image.fromarray(annotation.astype(dtype=np.uint8))
      pil_image.save(filename)

    def _convert_segmentation_to_raw(self, original_gt_folder, output_dir, segmentation_format='png'):
      # Create the output directory if not exists.
      if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
      #
      annotations = glob.glob(os.path.join(original_gt_folder, '*.' + segmentation_format))
      for annotation in annotations:
        raw_annotation = self._remove_colormap(annotation)
        filename = os.path.basename(annotation)[:-4]
        self._save_annotation(raw_annotation, os.path.join(output_dir, filename + '.' + segmentation_format))
      #

    def convert_dataset(self, root_folder):
        original_gt_folder = os.path.join(root_folder, 'SegmentationClass')
        output_dir = os.path.join(root_folder, 'SegmentationClassRaw')
        self._convert_segmentation_to_raw(original_gt_folder, output_dir)

