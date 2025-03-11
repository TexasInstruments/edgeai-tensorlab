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
import scipy.io
import sys
import glob
import random
import numpy as np
import cv2
import PIL
from colorama import Fore

from .. import utils
from .dataset_base import *

class NYUDepthV2(DatasetBase):
    def __init__(self, num_classes=151, ignore_label=None, download=False, num_frames=None, name='nyudepthv2', **kwargs):
        super().__init__(num_classes=num_classes, num_frames=num_frames, name=name, **kwargs)

        # moving this import hre, to make it conditional
        try:
            import h5py
        except:
            print(utils.log_color('WARNING', f'{os.path.basename(__file__)} - h5py package was not found, functionality in this file may not work'))
            raise

        self.force_download = True if download == 'always' else False
        assert 'path' in self.kwargs and 'split' in self.kwargs, 'path and split must be provided'

        self.depth_label_scale = 256.0
        path = self.kwargs['path']
        split = kwargs['split']
        if download:
            self.download(path, split)
        #

        self.name = "NYUDEPTHV2"
        self.ignore_label = ignore_label
        #self.label_dir_txt = os.path.join(self.kwargs['path'], 'objectInfo150.txt')

        image_dir = os.path.join(self.kwargs['path'], self.kwargs['split'], 'images')
        images_pattern = os.path.join(image_dir, '*.jpg')
        images = glob.glob(images_pattern)
        self.imgs = sorted(images)

        labels_dir = os.path.join(self.kwargs['path'], self.kwargs['split'], 'annotations')
        labels_pattern = os.path.join(labels_dir, '*.png')
        labels = glob.glob(labels_pattern)
        self.labels = sorted(labels)
        assert len(self.imgs) == len(self.labels), 'mismatch in the number f images and labels'

        self.num_frames = self.kwargs['num_frames'] = min(self.kwargs['num_frames'], len(self.imgs)) \
            if (self.kwargs['num_frames'] is not None) else len(self.imgs)

    def download(self, path, split):
        root = path
        out_folder = root
        train_images_folder = os.path.join(path, 'train', 'images')
        train_annotations_folder = os.path.join(path, 'train', 'annotations')
        val_images_folder = os.path.join(path, 'val', 'images')
        val_annotations_folder = os.path.join(path, 'val', 'annotations')
        if (not self.force_download) and os.path.exists(path) and os.path.exists(train_images_folder) and \
            os.path.exists(train_annotations_folder) and os.path.exists(val_images_folder) and \
            os.path.exists(val_annotations_folder):
            print(utils.log_color('\nINFO', 'dataset exists - will reuse', path))
            return
        #
        print(utils.log_color('\nINFO', 'downloading and preparing dataset', path + ' This may take some time.'))
        print(f'{Fore.YELLOW}'
              f'\nNYUDepthV2 Dataset:'
              f'\n    Indoor Segmentation and Support Inference from RGBD Images'
              f'\n       Silberman, N., Hoiem, D., Kohli, P., & Fergus, R. , European Conference on Computer Vision (ECCV), 2012. '
              f'\n    Visit the following urls to know more about NYUDepthV2 dataset: '            
              f'\n        https://www.tensorflow.org/datasets/catalog/nyu_depth_v2'
              f'\n        https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html '
              f'{Fore.RESET}\n')

        dataset_url = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
        split_url = 'https://github.com/cleinc/bts/blob/master/utils/splits.mat?raw=true'
        root = root.rstrip('/')
        download_root = os.path.join(root, 'download')
        file_path = utils.download_file(dataset_url, root=download_root, force_download=self.force_download)
        split_path = utils.download_file(split_url, root=download_root, force_download=self.force_download)

        h5_file = h5py.File(file_path, 'r')
        split = scipy.io.loadmat(split_path)

        os.makedirs(out_folder, exist_ok=True)
        os.makedirs(train_images_folder, exist_ok=True)
        os.makedirs(train_annotations_folder, exist_ok=True)
        os.makedirs(val_images_folder, exist_ok=True)
        os.makedirs(val_annotations_folder, exist_ok=True)

        test_images = set([int(x) for x in split["testNdxs"]])
        train_images = set([int(x) for x in split["trainNdxs"]])
        depths_raw = h5_file['rawDepths']

        images = h5_file['images']
        scenes = [u''.join(chr(c) for c in h5_file[obj_ref]) for obj_ref in h5_file['sceneTypes'][0]]

        for i, (image, scene, depth_raw) in enumerate(zip(images, scenes, depths_raw)):
            depth_raw = depth_raw.T
            image = image.T

            idx = int(i) + 1
            if idx in train_images:
                train_val = "train"
            else:
                assert idx in test_images, "index %d neither found in training set nor in test set" % idx
                train_val = "val"

            #folder = "%s/%s" % (out_folder, train_val)
            folder = os.path.join(out_folder, train_val)
            images_folder = os.path.join(folder, 'images')
            annotations_folder = os.path.join(folder, 'annotations')

            # if not os.path.exists(folder):
            #     os.makedirs(folder)

            depth_raw = depth_raw.clip(0.0, 255.0 )
            img_depth = depth_raw * self.depth_label_scale
            img_depth_uint16 = img_depth.astype(np.uint16)
            cv2.imwrite("%s/%05d.png" % (annotations_folder, i), img_depth_uint16)
            image = image[:, :, ::-1]
            image_black_boundary = np.zeros((480, 640, 3), dtype=np.uint8)
            image_black_boundary[7:474, 7:632, :] = image[7:474, 7:632, :]
            cv2.imwrite("%s/%05d.jpg" % (images_folder, i), image_black_boundary)
        #
        print(utils.log_color('\nINFO', 'dataset ready', path))
        return

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

    def __call__(self, predictions, **kwargs):
        return self.evaluate(predictions, **kwargs)

    def compute_scale_and_shift(self, prediction, gt, mask):
        a_00 = np.sum(mask * prediction * prediction)
        a_01 = np.sum(mask * prediction)
        a_11 = np.sum(mask)

        b_0 = np.sum(mask * prediction * gt)
        b_1 = np.sum(mask * gt)

        det = a_00 * a_11 - a_01 * a_01

        if det <= 0:
            return 0, 0

        else:
            x_0 = (a_11 * b_0 - a_01 * b_1) / det
            x_1 = (-a_01 * b_0 + a_00 * b_1) / det

            return x_0, x_1

    def evaluate(self, predictions, threshold=1.25, depth_cap_max = 80, depth_cap_min = 1e-3, **kwargs):
        disparity = kwargs.get('disparity')
        scale_and_shift_needed = kwargs.get('scale_shift')

        delta_1 = 0.0
        num_frames = min(self.num_frames, len(predictions))
        for n in range(num_frames):
            image_file, label_file = self.__getitem__(n, with_label=True)
            label_img = PIL.Image.open(label_file)
            label_img = np.array(label_img, dtype=np.float32) / self.depth_label_scale
            prediction = predictions[n]
            if scale_and_shift_needed:
                mask = label_img != 0
                disp_label = np.zeros_like(label_img)
                disp_label[mask] = 1.0 / label_img[mask]
                if not disparity:
                    disp_prediction = np.zeros_like(prediction)
                    disp_prediction[prediction != 0] = 1.0 / prediction[prediction != 0]
                else:
                    disp_prediction = prediction
                scale, shift = self.compute_scale_and_shift(disp_prediction, disp_label, mask)

                prediction = scale * disp_prediction + shift
                prediction[prediction < 1 / depth_cap_max] = 1 / depth_cap_max
                prediction[prediction > 1 / depth_cap_min] = 1 / depth_cap_min

            mask = np.minimum(label_img, prediction) != 0

            if disparity:
                disp_pred = prediction
                prediction = np.zeros_like(disp_pred)
                prediction[mask] = 1.0 / disp_pred[mask]

            delta = np.zeros_like(label_img, dtype=np.float32)
            delta = np.maximum(
                prediction[mask] / label_img[mask], 
                label_img[mask] / prediction[mask]
            )
            good_pixels_in_img = delta < threshold
            delta_1 += good_pixels_in_img.sum() / mask.sum()
        #

        delta_1 /= (n + 1)
        metric = {'accuracy_delta_1%': delta_1 * 100}
        return metric