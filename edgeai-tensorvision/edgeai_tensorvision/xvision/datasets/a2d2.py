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

'''
Reference:

A2D2: Audi Autonomous Driving Dataset,
Jakob Geyer, Yohannes Kassahun, Mentar Mahmudi, Xavier Ricou, Rupesh Durgesh, Andrew S. Chung,
Lorenz Hauswald, Viet Hoang Pham, Maximilian Mühlegg, Sebastian Dorn, Tiffany Fernandez, Martin Jänicke,
Sudesh Mirashi, Chiragkumar Savani, Martin Sturm, Oleksandr Vorobiov, Martin Oelker,
Sebastian Garreis, Peter Schuberth, https://arxiv.org/abs/2004.06320
'''

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
    dataset_config.split = 'val'
    return dataset_config


###########################################
class A2D2BaseSegmentationLoader():
    """A2D2Loader: Data is derived from A2D2, and can be downloaded from here: https://www.A2D2-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo: https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/A2D2.py"""
    
    train_only_difficult_classes = False
    class_weights_ = None

    colors = [
        [255,0,0],[182,89,6],[204,153,255],[255,128,0],[0,255,0],[0,128,255],[0,255,255],[255,255,0],[233,100,0],[110,110,0],[128,128,0],[255,193,37],[64,0,64],[185,122,87],[0,0,100],[139,99,108],[210,50,115],[255,0,128],[255,246,143],[150,0,150],[204,255,153],[238,162,173],[33,44,177],[180,50,180],[255,70,185],[238,233,191],[147,253,194],[150,150,200],[180,150,200],[72,209,204],[200,125,210],[159,121,238],[128,0,255],[255,0,255],[135,206,255],[241,230,255],[96,69,143],[53,46,82], [0, 0, 0]]

    num_classes_ = 38
    label_colours = dict(zip(range(num_classes_), colors))
    
    #Difficult classes less than < 0.4 mAP
    if train_only_difficult_classes:
        valid_classes = [8, 9, 11, 13, 14, 16, 17, 18, 19, 20, 22, 24, 25, 36, 37]
        void_classes = []
        #could not get pythonoic way working !!
        #void_classes = [x for x in range(num_classes_) if x in valid_classes]
        for idx in range(num_classes_):
            if idx not in valid_classes:
                void_classes.append(idx)
        
        print("void_classes: ", void_classes)
        colors_valid_class = []
        for idx, valid_class in enumerate(valid_classes):
            colors_valid_class.append(colors[valid_class]) 
        colors = colors_valid_class
    else:        
        valid_classes = range(0,num_classes_)
        void_classes = []
        # class_weights_ = np.ones(num_classes_)
        # #set high freq category weights to low to not over power other categorie
        # # Nature object 26
        # # RD normal street 33
        # # Sky 34
        # # Buildings 35

        # cat_with_high_freq = [26, 33, 34, 35]
        # for cat_idx in cat_with_high_freq:
        #     class_weights_[cat_idx] = 0.05

    num_valid_classes = len(valid_classes)
    class_names = ['Car  0','Bicycle  1','Pedestrian  2','Truck  3','Small vehicles  4','Traffic signal  5','Traffic sign  6','Utility vehicle  7','Sidebars 8','Speed bumper 9','Curbstone 10','Solid line 11','Irrelevant signs 12','Road blocks 13','Tractor 14','Non-drivable street 15','Zebra crossing 16','Obstacles / trash 17','Poles 18','RD restricted area 19','Animals 20','Grid structure 21','Signal corpus 22','Drivable cobbleston 23','Electronic traffic 24','Slow drive area 25','Nature object 26','Parking area 27','Sidewalk 28','Ego car 29','Painted driv. instr. 30','Traffic guide obj. 31','Dashed line 32','RD normal street 33','Sky 34','Buildings 35','Blurred area 36','Rain dirt 37']

    ignore_index = 255
    class_map = dict(zip(valid_classes, range(num_classes_)))

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
class A2D2BaseMotionLoader():
    """A2D2Loader: Data is derived from A2D2, and can be downloaded from here: https://www.A2D2-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo: https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/A2D2.py"""
    colors = [  # [  0,   0,   0],
        [0, 0, 0], [119, 11, 32]]

    label_colours = dict(zip(range(2), colors))

    void_classes = []
    valid_classes = [0, 255]
    class_names = ['static', 'moving']
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
class A2D2DataLoader(data.Dataset):
    def __init__(self, dataset_config, root, split="train", gt="gtFine", transforms=None, image_folders=('leftImg8bit',),
                 search_images=False, load_segmentation=True, load_depth=False, load_motion=False, load_flow=False,
                 load_segmentation_five_class=False, inference=False, additional_info=False, input_offsets=None):
        super().__init__()
        if split not in ['train', 'val', 'test', 'test_val']:
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

        #self.image_suffix = (self.image_folders[-1]+'.png') #'.png'
        #self.image_suffix = self.image_suffix.replace('leftImg8bit_sequence.png', 'leftImg8bit.png')
        self.image_suffix = '.png'
        #self.segmentation_suffix = self.gt+'_labelIds.png'  #'.png'
        self.segmentation_suffix = '.png'  #'.png'
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
            lbl = A2D2BaseSegmentationLoader.encode_segmap(np.array(lbl, dtype=np.uint8))
            targets.append(lbl)
            targets_path.append(segmentation_path)
        #

        elif self.load_segmentation_five_class and (not self.inference):
            lbl = cv2.imread(segmentation_path,0)
            lbl = A2D2BaseSegmentationLoaderFiveClasses.encode_segmap(np.array(lbl, dtype=np.uint8))
            targets.append(lbl)
            targets_path.append(segmentation_path)

        if self.load_motion and (not self.inference):
            motion_path = image_path.replace(self.image_base, self.motion_base).replace(self.image_suffix, self.motion_suffix)
            self.check_file_exists(motion_path)
            motion = cv2.imread(motion_path,0)
            motion = A2D2BaseMotionLoader.encode_segmap(np.array(motion, dtype=np.uint8))
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
            return A2D2BaseSegmentationLoader.decode_segmap(lbl)
        elif self.load_segmentation_five_class:
            return A2D2BaseSegmentationLoaderFiveClasses.decode_segmap(lbl)
        else:
            return A2D2BaseMotionLoader.decode_segmap(lbl)
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
            nc.append(A2D2BaseSegmentationLoader.num_classes_)
        elif self.load_segmentation_five_class:
            nc.append(A2D2BaseSegmentationLoaderFiveClasses.num_classes_)
        if self.load_motion:
            nc.append(A2D2BaseMotionLoader.num_classes_)
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
            cw.append(A2D2BaseSegmentationLoader.class_weights())
        elif self.load_segmentation_five_class:
            cw.append(A2D2BaseSegmentationLoaderFiveClasses.class_weights())
        if self.load_motion:
            cw.append(A2D2BaseMotionLoader.class_weights())
        #
        return cw
    #


    def create_palette(self):
        palette = []
        if self.load_segmentation:
            palette.append(A2D2BaseSegmentationLoader.colors)
        if self.load_segmentation_five_class:
            palette.append(A2D2BaseSegmentationLoaderFiveClasses.colors)
        if self.load_motion:
            palette.append(A2D2BaseMotionLoader.colors)
        return palette


##########################################
def a2d2_segmentation_train(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    transform = transforms[0] if isinstance(transforms, (list,tuple)) else transforms
    train_split = A2D2DataLoader(dataset_config, root, 'train', gt, transforms=transform,
                                            load_segmentation=dataset_config.load_segmentation,
                                            load_segmentation_five_class=dataset_config.load_segmentation_five_class)
    return train_split


def a2d2_segmentation(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    train_split = val_split = None
    if split is None:
        split = ['train', 'val']
    for split_name in split:
        if split_name == 'train':
            train_split = A2D2DataLoader(dataset_config, root, split_name, gt, transforms=transforms[0],
                                            load_segmentation=dataset_config.load_segmentation,
                                            load_segmentation_five_class=dataset_config.load_segmentation_five_class)
        else:
            val_split = A2D2DataLoader(dataset_config, root, split_name, gt, transforms=transforms[1],
                                            load_segmentation=dataset_config.load_segmentation,
                                            load_segmentation_five_class=dataset_config.load_segmentation_five_class)
    #
    return train_split, val_split


def a2d2_depth_train(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    train_split = A2D2DataLoader(dataset_config, root, 'train', gt, transforms=transforms[0], load_segmentation=False, load_depth = True)
    return train_split


def a2d2_depth(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    train_split = val_split = None
    split = ['train', 'val']
    for split_name in split:
        if split_name == 'train':
            train_split = A2D2DataLoader(dataset_config, root, split_name, gt, transforms=transforms[0], load_segmentation=False, load_depth = True)
        elif split_name == 'val':
            val_split = A2D2DataLoader(dataset_config, root, split_name, gt, transforms=transforms[1], load_segmentation=False, load_depth = True)
        else:
            pass
    #
    return train_split, val_split


#################################################################
# semantic inference
def a2d2_segmentation_infer(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    split_name = dataset_config.split #'val'
    infer_split = A2D2DataLoader(dataset_config, root, split_name, gt, transforms=transforms, image_folders=dataset_config.image_folders,
                                       load_segmentation=dataset_config.load_segmentation,
                                       load_segmentation_five_class=dataset_config.load_segmentation_five_class,
                                       search_images=True, inference=True, additional_info=True)
    return  infer_split


def a2d2_segmentation_measure(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    split_name = dataset_config.split #'val'
    infer_split = A2D2DataLoader(dataset_config, root, split_name, gt, transforms=transforms, image_folders=dataset_config.image_folders,
                                       load_segmentation=dataset_config.load_segmentation,
                                       load_segmentation_five_class=dataset_config.load_segmentation_five_class,
                                       search_images=True, inference=False, additional_info=True)
    return infer_split

def a2d2_segmentation_infer_dir(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    split_name = 'val'
    infer_split = A2D2DataLoader(dataset_config, root, split_name, gt, transforms=transforms, image_folders=dataset_config.image_folders,
                                       load_segmentation=dataset_config.load_segmentation,
                                       load_segmentation_five_class=dataset_config.load_segmentation_five_class,
                                       search_images=True, inference=True, additional_info=True)
    return infer_split


#############################################################################################################
# dual stream ip

def a2d2_segmentation_multi_input(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    train_split = val_split = None
    split = ['train', 'val']
    for split_name in split:
        if split_name == 'train':
            train_split = A2D2DataLoader(dataset_config, root, split_name, gt, transforms=transforms[0],
                                               image_folders=dataset_config.image_folders, input_offsets=dataset_config.input_offsets,
                                               load_segmentation=dataset_config.load_segmentation,
                                               load_segmentation_five_class=dataset_config.load_segmentation_five_class)
        elif split_name == 'val':
            val_split = A2D2DataLoader(dataset_config, root, split_name, gt, transforms=transforms[1],
                                             image_folders=dataset_config.image_folders, input_offsets=dataset_config.input_offsets,
                                             load_segmentation=dataset_config.load_segmentation,
                                             load_segmentation_five_class=dataset_config.load_segmentation_five_class)
        else:
            pass
    #
    return train_split, val_split

def a2d2_motion_multi_input(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    train_split = val_split = None
    split = ['train', 'val']
    for split_name in split:
        if split_name == 'train':
            train_split = A2D2DataLoader(dataset_config, root, split_name, gt, transforms=transforms[0], load_segmentation=False, load_motion = True,
                                               image_folders=dataset_config.image_folders, input_offsets=dataset_config.input_offsets)
        elif split_name == 'val':
            val_split = A2D2DataLoader(dataset_config, root, split_name, gt, transforms=transforms[1], load_segmentation=False, load_motion = True,
                                             image_folders=dataset_config.image_folders, input_offsets=dataset_config.input_offsets)
        else:
            pass
    #
    return train_split, val_split


def a2d2_depth_semantic_motion_multi_input(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    train_split = val_split = None
    split = ['train', 'val']
    for split_name in split:
        if split_name == 'train':
            train_split = A2D2DataLoader(dataset_config, root, split_name, gt, transforms=transforms[0], load_depth = True,
                                               load_motion=True, image_folders=dataset_config.image_folders, input_offsets=dataset_config.input_offsets,
                                               load_segmentation=dataset_config.load_segmentation,
                                               load_segmentation_five_class=dataset_config.load_segmentation_five_class)
        elif split_name == 'val':
            val_split = A2D2DataLoader(dataset_config, root, split_name, gt, transforms=transforms[1], load_depth = True,
                                             load_motion=True, image_folders=dataset_config.image_folders, input_offsets=dataset_config.input_offsets,
                                             load_segmentation=dataset_config.load_segmentation,
                                             load_segmentation_five_class=dataset_config.load_segmentation_five_class)
        else:
            pass
    #
    return train_split, val_split

#############################################################################################################
# inference dual stream
def a2d2_segmentation_multi_input_measure(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    split_name = 'val'
    infer_split = A2D2DataLoader(dataset_config, root, split_name, gt, transforms=transforms, image_folders=dataset_config.image_folders,
                                       load_segmentation=dataset_config.load_segmentation,
                                       load_segmentation_five_class=dataset_config.load_segmentation_five_class,
                                       search_images=True, inference=False, additional_info=True, input_offsets=dataset_config.input_offsets)
    return infer_split



# motion inference
def a2d2_motion_multi_input_infer(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    dataset_config.image_folders = ('leftImg8bit_flow_confidence', 'leftImg8bit')
    gt = "gtFine"
    split_name = 'val'
    val_split = A2D2DataLoader(dataset_config, root, split_name, gt, transforms=transforms, load_segmentation=False,
                                     image_folders=dataset_config.image_folders, search_images=True, inference=True, additional_info=True,
                                     input_offsets=dataset_config.input_offsets)
    #
    return val_split


def a2d2_motion_multi_input_measure(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    split_name = 'val'
    val_split = A2D2DataLoader(dataset_config, root, split_name, gt, transforms=transforms, load_segmentation=False, load_motion = True,
                                     image_folders=dataset_config.image_folders, search_images=True, inference=False, additional_info=True,
                                     input_offsets=dataset_config.input_offsets)
    #
    return val_split


def a2d2_depth_semantic_motion_multi_input_infer(dataset_config, root, split=None, transforms=None):
    dataset_config = get_config().merge_from(dataset_config)
    gt = "gtFine"
    split_name = 'val'
    val_split = A2D2DataLoader(dataset_config, root, split_name, gt, transforms=transforms, load_depth = True,
                                     load_motion=True, image_folders=dataset_config.image_folders,search_images=True,
                                     inference=True, additional_info=True, input_offsets = dataset_config.input_offsets)
    #
    return val_split
