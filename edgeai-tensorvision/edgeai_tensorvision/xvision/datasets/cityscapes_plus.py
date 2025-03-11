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

# ==============================================================================

# Some parts of the code are borrowed from: https://github.com/ansleliu/LightNet
# with the following license:
#
# MIT License
#
# Copyright (c) 2018 Huijun Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Reference:

M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele,
“The Cityscapes Dataset for Semantic Urban Scene Understanding,”
in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
https://www.cityscapes-dataset.com/
"""


import os
import numpy as np
import cv2
import json
from torch.utils import data
import sys
import warnings
from edgeai_torchmodelopt import xnn

###########################################
# config settings
def get_config():
    dataset_config = xnn.utils.ConfigNode()
    dataset_config.image_folders = ('leftImg8bit',)
    dataset_config.input_offsets = None
    dataset_config.load_segmentation = True
    dataset_config.load_segmentation_five_class = False
    return dataset_config


###########################################
class CityscapesBaseSegmentationLoader():
    """CityscapesLoader: Data is derived from CityScapes, and can be downloaded from here: https://www.cityscapes-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo: https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py"""

    colors = [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
        [153, 153, 153],[250, 170, 30], [220, 220, 0],[107, 142, 35],  [152, 251, 152],
        [0, 130, 180],  [220, 20, 60],  [255, 0, 0],  [0, 0, 142],     [0, 0, 70],
        [0, 60, 100],   [0, 80, 100],   [0, 0, 230],  [119, 11, 32], [0, 0, 0]]

    label_colours = dict(zip(range(19), colors))
    
    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    # class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
    #                     'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
    #                     'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    #                     'motorcycle', 'bicycle']

    ignore_index = 255
    class_map = dict(zip(valid_classes, range(19)))
    num_classes_ = 19

    class_weights_ = np.array([0.05570516, 0.32337477, 0.08998544, 1.03602707, 1.03413147, 1.68195437,
                               5.58540548, 3.56563995, 0.12704978, 1., 0.46783719, 1.34551528,
                               5.29974114, 0.28342531, 0.9396095, 0.81551811, 0.42679146, 3.6399074,
                               2.78376194], dtype=float)

    @classmethod
    def decode_segmap(cls, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, cls.num_classes_):
            r[temp == l] = cls.label_colours[l][0]
            g[temp == l] = cls.label_colours[l][1]
            b[temp == l] = cls.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb


    @classmethod
    def encode_segmap(cls, mask):
        # Put all void classes to zero
        for _voidc in cls.void_classes:
            mask[mask == _voidc] = cls.ignore_index
        for _validc in cls.valid_classes:
            mask[mask == _validc] = cls.class_map[_validc]
        return mask


    @classmethod
    def class_weights(cls):
        return cls.class_weights_


###########################################
class CityscapesBaseSegmentationLoaderFiveClasses():
    """CityscapesLoader: Data is derived from CityScapes, and can be downloaded from here: https://www.cityscapes-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo: https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py"""

    colors = [  # [  0,   0,   0],
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [0, 0, 0]]
    label_colours = dict(zip(range(5), colors))
    void_classes = [-1, 255]
    valid_classes = [0, 1, 2, 3, 4]
    # class_names = ['road', 'sky', 'pedestrian', 'vehicle', 'background']

    ignore_index = 255
    class_map = dict(zip(valid_classes, range(5)))
    num_classes_ = 5

    class_weights_ = np.array([0.22567085, 1.89944273, 5.24032014, 1., 0.13516443], dtype=float)

    @classmethod
    def decode_segmap(cls, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, cls.num_classes_):
            r[temp == l] = cls.label_colours[l][0]
            g[temp == l] = cls.label_colours[l][1]
            b[temp == l] = cls.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    @classmethod
    def encode_segmap(cls, mask):
        # Put all void classes to zero
        for _voidc in cls.void_classes:
            mask[mask == _voidc] = cls.ignore_index
        for _validc in cls.valid_classes:
            mask[mask == _validc] = cls.class_map[_validc]
        return mask

    @classmethod
    def class_weights(cls):
        return cls.class_weights_


###########################################
class CityscapesBaseMotionLoader():
    """CityscapesLoader: Data is derived from CityScapes, and can be downloaded from here: https://www.cityscapes-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo: https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py"""
    colors = [  # [  0,   0,   0],
        [0, 0, 0], [119, 11, 32]]

    label_colours = dict(zip(range(2), colors))

    void_classes = []
    valid_classes = [0, 255]
    # class_names = ['static', 'moving']
    ignore_index = 255
    class_map = dict(zip(valid_classes, range(2)))
    num_classes_ = 2
    class_weights_ = np.array([0.05, 0.95], dtype=float)    #Calculated weights based on mdeian_frequenncy = [ 0.51520306, 16.94405377]

    @classmethod
    def decode_segmap(cls, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, cls.num_classes_):
            r[temp == l] = cls.label_colours[l][0]
            g[temp == l] = cls.label_colours[l][1]
            b[temp == l] = cls.label_colours[l][2]
        #
        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    @classmethod
    def encode_segmap(cls, mask):
        for _validc in cls.valid_classes:
            mask[mask == _validc] = cls.class_map[_validc]
        # Put all void classes to zero
        for _voidc in cls.void_classes:
            mask[mask == _voidc] = cls.ignore_index
        return mask

    @classmethod
    def class_weights(cls):
        return cls.class_weights_


###########################################
class CityscapesDataLoader(data.Dataset):
    def __init__(self, dataset_config, root, split="train", gt="gtFine", transforms=None, image_folders=('leftImg8bit',),
                 search_images=False, load_segmentation=True, load_depth=False, load_motion=False, load_flow=False,
                 load_segmentation_five_class=False, inference=False, additional_info=False, input_offsets=None, annotation_prefix=None):
        super().__init__()
        if split not in ['train', 'val', 'test']:
            warnings.warn(f'unknown split specified: {split}')
        #
        self.root = root
        self.gt = gt
        self.split = split
        self.transforms = transforms
        self.image_folders = image_folders
        self.search_images = search_images
        self.files = {}

        self.additional_info = additional_info
        self.load_segmentation = load_segmentation
        self.load_segmentation_five_class = load_segmentation_five_class
        self.load_depth = load_depth
        self.load_motion = load_motion
        self.load_flow = load_flow
        self.inference = inference
        self.input_offsets = input_offsets

        self.image_suffix = (self.image_folders[-1]+'.png') #'.png'
        self.image_suffix = self.image_suffix.replace('leftImg8bit_sequence.png', 'leftImg8bit.png')
        self.segmentation_suffix = self.gt+'_labelIds.png'  #'.png'
        if self.load_segmentation_five_class:
            self.segmentation_suffix = self.gt+'_labelTrainIds.png'
        self.disparity_suffix = 'disparity.png'
        self.motion_suffix =  self.gt+'_labelTrainIds_motion.png' #'.png'

        self.image_base = os.path.join(self.root, image_folders[-1], self.split)
        self.segmentation_base = os.path.join(self.root, gt, self.split)
        self.disparity_base = os.path.join(self.root, 'disparity', self.split)
        self.cameracalib_base = os.path.join(self.root, 'camera', self.split)
        self.motion_base = os.path.join(self.root, gt, self.split)

        if self.search_images:
            self.files = xnn.utils.recursive_glob(rootdir=self.image_base, suffix=self.image_suffix)
        else:
            self.files = xnn.utils.recursive_glob(rootdir=self.segmentation_base, suffix=self.segmentation_suffix)
        #
        self.files = sorted(self.files)
        
        if not self.files:
            raise Exception("> No files for split=[%s] found in %s" % (split, self.segmentation_base))
        #
        
        self.image_files = [None] * len(image_folders)
        for image_idx, image_folder in enumerate(image_folders):
            image_base = os.path.join(self.root, image_folder, self.split)
            self.image_files[image_idx] = sorted(xnn.utils.recursive_glob(rootdir=image_base, suffix='.png'))
            assert len(self.image_files[image_idx]) == len(self.image_files[0]), 'all folders should have same number of files'
        #
        
        
    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        if self.search_images:
            image_path = self.files[index].rstrip()
            self.check_file_exists(image_path)
            segmentation_path = image_path.replace(self.image_base, self.segmentation_base).replace(self.image_suffix, self.segmentation_suffix)
        else:
            segmentation_path = self.files[index].rstrip()
            self.check_file_exists(segmentation_path)
            image_path = segmentation_path.replace(self.segmentation_base, self.image_base).replace(self.segmentation_suffix, self.image_suffix)
        #

        images = []
        images_path = []
        for image_idx, image_folder in enumerate(self.image_folders):
            sys.stdout.flush()
            this_image_path =  self.image_files[image_idx][index].rstrip()
            if image_idx == (len(self.image_folders)-1):
                assert this_image_path == image_path, 'image file name error'
            #
            self.check_file_exists(this_image_path)

            img = cv2.imread(this_image_path)[:,:,::-1]
            if self.input_offsets is not None:
                img = img - self.input_offsets[image_idx]
            #
            images.append(img)
            images_path.append(this_image_path)
        #

        targets = []
        targets_path = []
        if self.load_flow and (not self.inference):
            flow_zero = np.zeros((images[0].shape[0],images[0].shape[1],2), dtype=np.float32)
            targets.append(flow_zero)

        if self.load_depth and (not self.inference):
            disparity_path = image_path.replace(self.image_base, self.disparity_base).replace(self.image_suffix, self.disparity_suffix)
            self.check_file_exists(disparity_path)
            depth = self.depth_loader(disparity_path)
            targets.append(depth)
        #

        if self.load_segmentation and (not self.inference):
            lbl = cv2.imread(segmentation_path,0)
            lbl = CityscapesBaseSegmentationLoader.encode_segmap(np.array(lbl, dtype=np.uint8))
            targets.append(lbl)
            targets_path.append(segmentation_path)
        #

        elif self.load_segmentation_five_class and (not self.inference):
            lbl = cv2.imread(segmentation_path,0)
            lbl = CityscapesBaseSegmentationLoaderFiveClasses.encode_segmap(np.array(lbl, dtype=np.uint8))
            targets.append(lbl)
            targets_path.append(segmentation_path)

        if self.load_motion and (not self.inference):
            motion_path = image_path.replace(self.image_base, self.motion_base).replace(self.image_suffix, self.motion_suffix)
            self.check_file_exists(motion_path)
            motion = cv2.imread(motion_path,0)
            motion = CityscapesBaseMotionLoader.encode_segmap(np.array(motion, dtype=np.uint8))
            targets.append(motion)
        #

        #targets = np.stack(targets, axis=2)

        if (self.transforms is not None):
            images, targets = self.transforms(images, targets)
        #

        if self.additional_info:
            return images, targets, images_path, targets_path
        else:
            return images, targets
    #


    def decode_segmap(self, lbl):
        if self.load_segmentation:
            return CityscapesBaseSegmentationLoader.decode_segmap(lbl)
        elif self.load_segmentation_five_class:
            return CityscapesBaseSegmentationLoaderFiveClasses.decode_segmap(lbl)
        else:
            return CityscapesBaseMotionLoader.decode_segmap(lbl)
    #


    def check_file_exists(self, file_name):
        if not os.path.exists(file_name) or not os.path.isfile(file_name):
            raise Exception("{} is not a file, can not open with imread.".format(file_name))
    #


    def depth_loader(self, disparity_path):
        eps = (1e-6)
        disparity_range = (eps, 255.0)
        depth_range = (1.0, 255.0)

        disp = cv2.imread(disparity_path, cv2.IMREAD_UNCHANGED)
        disp_valid = (disp > 0)       # 0 values have to be ignored
        disp = ((disp - 1.0)/256.0)

        # convert to depth
        calib_path = disparity_path.replace(self.disparity_base, self.cameracalib_base).replace(self.disparity_suffix, 'camera.json')
        with open(calib_path) as fp:
            cameracalib = json.load(fp)
            extrinsic = cameracalib['extrinsic']
            intrinsic = cameracalib['intrinsic']
            focal_len = intrinsic['fx']
            proj = (focal_len * extrinsic['baseline'])
            depth = np.divide(proj, disp, out=np.zeros_like(disp), where=(disp!=0))
            d_out = np.clip(depth, depth_range[0], depth_range[1]) * disp_valid
        #
        return d_out
    #


    def num_classes(self):
        nc = []
        if self.load_flow:
            nc.append(2)
        if self.load_depth:
            nc.append(1)
        if self.load_segmentation:
            nc.append(CityscapesBaseSegmentationLoader.num_classes_)
        elif self.load_segmentation_five_class:
            nc.append(CityscapesBaseSegmentationLoaderFiveClasses.num_classes_)
        if self.load_motion:
            nc.append(CityscapesBaseMotionLoader.num_classes_)
        #
        return nc
    #


    def class_weights(self):
        cw = []
        if self.load_flow:
            cw.append(None)
        if self.load_depth:
            cw.append(None)
        if self.load_segmentation:
            cw.append(CityscapesBaseSegmentationLoader.class_weights())
        elif self.load_segmentation_five_class:
            cw.append(CityscapesBaseSegmentationLoaderFiveClasses.class_weights())
        if self.load_motion:
            cw.append(CityscapesBaseMotionLoader.class_weights())
        #
        return cw
    #


    def create_palette(self):
        palette = []
        if self.load_segmentation:
            palette.append(CityscapesBaseSegmentationLoader.colors)
        if self.load_segmentation_five_class:
            palette.append(CityscapesBaseSegmentationLoaderFiveClasses.colors)
        if self.load_motion:
            palette.append(CityscapesBaseMotionLoader.colors)
        return palette


##########################################
def cityscapes_segmentation_train(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    transform = transforms[0] if isinstance(transforms, (list,tuple)) else transforms
    train_split = CityscapesDataLoader(dataset_config, root, 'train', gt, transforms=transform,
                                            load_segmentation=dataset_config.load_segmentation,
                                            load_segmentation_five_class=dataset_config.load_segmentation_five_class)
    return train_split


def cityscapes_segmentation(dataset_config, root, split=None, transforms=None, *args, **kwargs):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    train_split = val_split = None
    split = ['train', 'val']
    for split_name in split:
        if split_name == 'train':
            train_split = CityscapesDataLoader(dataset_config, root, split_name, gt, transforms=transforms[0],
                                            load_segmentation=dataset_config.load_segmentation,
                                            load_segmentation_five_class=dataset_config.load_segmentation_five_class,
                                            *args, **kwargs)
        elif split_name == 'val':
            val_split = CityscapesDataLoader(dataset_config, root, split_name, gt, transforms=transforms[1],
                                            load_segmentation=dataset_config.load_segmentation,
                                            load_segmentation_five_class=dataset_config.load_segmentation_five_class,
                                            *args, **kwargs)
        else:
            pass
    #
    return train_split, val_split


def cityscapes_segmentation_with_additional_info(dataset_config, root, split=None, transforms=None, *args, **kwargs):
    return cityscapes_segmentation(dataset_config, root, split=None, transforms=transforms, additional_info=True, *args, **kwargs)


def cityscapes_depth_train(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    train_split = CityscapesDataLoader(dataset_config, root, 'train', gt, transforms=transforms[0], load_segmentation=False, load_depth = True)
    return train_split


def cityscapes_depth(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    train_split = val_split = None
    split = ['train', 'val']
    for split_name in split:
        if split_name == 'train':
            train_split = CityscapesDataLoader(dataset_config, root, split_name, gt, transforms=transforms[0], load_segmentation=False, load_depth = True)
        elif split_name == 'val':
            val_split = CityscapesDataLoader(dataset_config, root, split_name, gt, transforms=transforms[1], load_segmentation=False, load_depth = True)
        else:
            pass
    #
    return train_split, val_split


#################################################################
# semantic inference
def cityscapes_segmentation_infer(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    split_name = 'val'
    infer_split = CityscapesDataLoader(dataset_config, root, split_name, gt, transforms=transforms, image_folders=dataset_config.image_folders,
                                       load_segmentation=dataset_config.load_segmentation,
                                       load_segmentation_five_class=dataset_config.load_segmentation_five_class,
                                       search_images=True, inference=True, additional_info=True)
    return  infer_split


def cityscapes_segmentation_measure(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    split_name = 'val'
    infer_split = CityscapesDataLoader(dataset_config, root, split_name, gt, transforms=transforms, image_folders=dataset_config.image_folders,
                                       load_segmentation=dataset_config.load_segmentation,
                                       load_segmentation_five_class=dataset_config.load_segmentation_five_class,
                                       search_images=True, inference=False, additional_info=True)
    return infer_split

def cityscapes_segmentation_infer_dir(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    split_name = 'val'
    infer_split = CityscapesDataLoader(dataset_config, root, split_name, gt, transforms=transforms, image_folders=dataset_config.image_folders,
                                       load_segmentation=dataset_config.load_segmentation,
                                       load_segmentation_five_class=dataset_config.load_segmentation_five_class,
                                       search_images=True, inference=True, additional_info=True)
    return infer_split


#############################################################################################################
# dual stream ip

def cityscapes_segmentation_multi_input(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    train_split = val_split = None
    split = ['train', 'val']
    for split_name in split:
        if split_name == 'train':
            train_split = CityscapesDataLoader(dataset_config, root, split_name, gt, transforms=transforms[0],
                                               image_folders=dataset_config.image_folders, input_offsets=dataset_config.input_offsets,
                                               load_segmentation=dataset_config.load_segmentation,
                                               load_segmentation_five_class=dataset_config.load_segmentation_five_class)
        elif split_name == 'val':
            val_split = CityscapesDataLoader(dataset_config, root, split_name, gt, transforms=transforms[1],
                                             image_folders=dataset_config.image_folders, input_offsets=dataset_config.input_offsets,
                                             load_segmentation=dataset_config.load_segmentation,
                                             load_segmentation_five_class=dataset_config.load_segmentation_five_class)
        else:
            pass
    #
    return train_split, val_split

def cityscapes_motion_multi_input(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    train_split = val_split = None
    split = ['train', 'val']
    for split_name in split:
        if split_name == 'train':
            train_split = CityscapesDataLoader(dataset_config, root, split_name, gt, transforms=transforms[0], load_segmentation=False, load_motion = True,
                                               image_folders=dataset_config.image_folders, input_offsets=dataset_config.input_offsets)
        elif split_name == 'val':
            val_split = CityscapesDataLoader(dataset_config, root, split_name, gt, transforms=transforms[1], load_segmentation=False, load_motion = True,
                                             image_folders=dataset_config.image_folders, input_offsets=dataset_config.input_offsets)
        else:
            pass
    #
    return train_split, val_split


def cityscapes_depth_semantic_motion_multi_input(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    train_split = val_split = None
    split = ['train', 'val']
    for split_name in split:
        if split_name == 'train':
            train_split = CityscapesDataLoader(dataset_config, root, split_name, gt, transforms=transforms[0], load_depth = True,
                                               load_motion=True, image_folders=dataset_config.image_folders, input_offsets=dataset_config.input_offsets,
                                               load_segmentation=dataset_config.load_segmentation,
                                               load_segmentation_five_class=dataset_config.load_segmentation_five_class)
        elif split_name == 'val':
            val_split = CityscapesDataLoader(dataset_config, root, split_name, gt, transforms=transforms[1], load_depth = True,
                                             load_motion=True, image_folders=dataset_config.image_folders, input_offsets=dataset_config.input_offsets,
                                             load_segmentation=dataset_config.load_segmentation,
                                             load_segmentation_five_class=dataset_config.load_segmentation_five_class)
        else:
            pass
    #
    return train_split, val_split

#############################################################################################################
# inference dual stream
def cityscapes_segmentation_multi_input_measure(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    split_name = 'val'
    infer_split = CityscapesDataLoader(dataset_config, root, split_name, gt, transforms=transforms, image_folders=dataset_config.image_folders,
                                       load_segmentation=dataset_config.load_segmentation,
                                       load_segmentation_five_class=dataset_config.load_segmentation_five_class,
                                       search_images=True, inference=False, additional_info=True, input_offsets=dataset_config.input_offsets)
    return infer_split



# motion inference
def cityscapes_motion_multi_input_infer(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    dataset_config.image_folders = ('leftImg8bit_flow_confidence', 'leftImg8bit')
    gt = "gtFine"
    split_name = 'val'
    val_split = CityscapesDataLoader(dataset_config, root, split_name, gt, transforms=transforms, load_segmentation=False,
                                     image_folders=dataset_config.image_folders, search_images=True, inference=True, additional_info=True,
                                     input_offsets=dataset_config.input_offsets)
    #
    return val_split


def cityscapes_motion_multi_input_measure(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    split_name = 'val'
    val_split = CityscapesDataLoader(dataset_config, root, split_name, gt, transforms=transforms, load_segmentation=False, load_motion = True,
                                     image_folders=dataset_config.image_folders, search_images=True, inference=False, additional_info=True,
                                     input_offsets=dataset_config.input_offsets)
    #
    return val_split


def cityscapes_depth_semantic_motion_multi_input_infer(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    split_name = 'val'
    val_split = CityscapesDataLoader(dataset_config, root, split_name, gt, transforms=transforms, load_depth = True,
                                     load_motion=True, image_folders=dataset_config.image_folders,search_images=True,
                                     inference=True, additional_info=True, input_offsets = dataset_config.input_offsets)
    #
    return val_split
