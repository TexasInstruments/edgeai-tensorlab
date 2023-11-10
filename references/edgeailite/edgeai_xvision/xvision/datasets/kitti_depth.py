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
Reference: http://www.cvlibs.net/datasets/kitti/

Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite,
Andreas Geiger and Philip Lenz and Raquel Urtasun,
Conference on Computer Vision and Pattern Recognition (CVPR), 2012

Vision meets Robotics: The KITTI Dataset,
Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun,
International Journal of Robotics Research (IJRR), 2013

A New Performance Measure and Evaluation Benchmark for Road Detection Algorithms,
Jannik Fritsch and Tobias Kuehnl and Andreas Geiger,
International Conference on Intelligent Transportation Systems (ITSC), 2013

Object Scene Flow for Autonomous Vehicles,
Moritz Menze and Andreas Geiger,
Conference on Computer Vision and Pattern Recognition (CVPR), 2015
"""

from __future__ import division
import os.path
import glob
from .dataset_utils import split2list, ListDataset, ListDatasetWithAdditionalInfo
import cv2
import numpy as np
import warnings
from edgeai_torchmodelopt import xnn

__all__ = ['kitti_depth','kitti_depth2','kitti_depth_sceneflow', 'kitti_depth_infer']

'''
Dataset routines for kitti_depth
http://www.cvlibs.net/datasets/kitti/eval_depth_all.php
'''

#load one image, one target depth
def kitti_depth(dataset_config, root, split, transforms):
    return kitti_depth_factory(root, split, transforms, loader=kitti_depth_loader, num_images=1, num_targets=1)


#load two images and one target depth
def kitti_depth2(dataset_config, root, split, transforms):
    return kitti_depth_factory(root, split, transforms, loader=kitti_depth_loader, num_images=2, num_targets=1)


#load two images, one target depth and insert two blank images for flow target
def kitti_depth_sceneflow(dataset_config, root, split=None, transforms=None):
    print('{}=> kitti_depth_sceneflow dataset has empty flow. Sparse estimation mode may be needed to ignore it {}'
          .format(xnn.utils.TermColor.YELLOW, xnn.utils.TermColor.END))
    return kitti_depth_factory(root, split, transforms, loader=kitti_depth_sceneflow_loader, num_images=2, num_targets=1)
def kitti_depth_infer(dataset_config, root, split, transforms):
    if isinstance(split, (list,tuple)) and len(split)>1 and isinstance(transforms, (list,tuple)) and len(transforms)>1:
        return kitti_depth(dataset_config, root, split, transforms)
    else:
        split = [split, split] if not isinstance(split, (list,tuple)) else split
        transforms = [transforms, transforms] if not isinstance(transforms, (list,tuple)) else transforms
        datasets =  kitti_depth_factory_with_additional_info(root, split, transforms, loader=kitti_depth_loader, num_images=1, num_targets=1)
        return datasets[1]


############################################################################
# internal functions
############################################################################
def load_depth_from_png(png_path):
    flo_file = cv2.imread(png_path,cv2.IMREAD_UNCHANGED)
    flo_file = flo_file[...,np.newaxis]
    flo_img = flo_file.astype(np.float32)
    invalid = (flo_file == 0)
    flo_img = flo_img / 256

    #indicate invalid regions by 0
    eps = 1e-16
    meps = -eps
    flo_img[(flo_img<eps) & (flo_img>=0)] = eps
    flo_img[(flo_img>meps) & (flo_img<0)] = meps
    flo_img[invalid] = 0

    return(flo_img)


def search_dataset_folder(dir, cam_folder, num_images, num_targets, extra_images_offset=None, depth_folder=''):
    '''Will search in the given folder'''

    image_folder = os.path.join(dir, cam_folder)
    depth_folder = os.path.join(dir, depth_folder)
    image_list = sorted(glob.glob(os.path.join(image_folder, '*.png')))
    depth_list = sorted(glob.glob(os.path.join(depth_folder, '*.png')))

    if len(image_list) != (len(depth_list) + 2*extra_images_offset):
        print('=> ignoring dir({})\n   num images({}) != num_depth({}) + {}'.format(image_folder, len(image_list), len(depth_list), 2*extra_images_offset))
        return []
    #

    #trim the image list to match the target list
    if extra_images_offset:
        image_list = image_list[extra_images_offset:-extra_images_offset+1]

    entries = []
    for i in range(len(depth_list)-1):
        input_list = []
        target_list = []
        for img_idx in range(num_images):
            img = image_list[i+img_idx]
            input_list.append(img)
        #
        for tgt_idx in range(num_targets):
            tgt = depth_list[i + tgt_idx]
            target_list.append(tgt)
        #
        entries.append((input_list, target_list))
    #
    return entries


def search_dataset_folders(folders, cam_list=None, num_images=None, num_targets=None, extra_images_offset=None, depth_list=None):
    images_list = []
    for folder in folders:
        for cam_folder, depth_folder in zip(cam_list, depth_list):
            images_folder = search_dataset_folder(folder.strip(), cam_folder, num_images=num_images, num_targets=num_targets,
                                                  extra_images_offset=extra_images_offset, depth_folder=depth_folder)
            images_list.extend(images_folder)
    return images_list


def make_dataset_splits(dir, split, num_images, num_targets):
    '''Will search in dataset folder'''
    if isinstance(split, (list,tuple)) and len(split)>1:
        with open(split[0]) as f:
            train_folders = [os.path.join(dir,f) for f in f.readlines()]
        with open(split[1]) as f:
            test_folders = [os.path.join(dir,f) for f in f.readlines()]
    elif isinstance(split, float):
        folders = glob.glob(os.path.join(dir,'*/*'))
        folders = [f for f in folders if os.path.isdir(f)]
        train_folders, test_folders = split2list(folders, split)
    elif isinstance(split, str):
        folders = glob.glob(os.path.join(dir,'*/*'))
        folders = [f for f in folders if os.path.isdir(f)]
        train_folders, test_folders = folders, folders
        warnings.warn('both splits returned are same, since the split was a string')
    else:
        assert False, 'split could not be understood'

    cam_list = ['image_02/data'] #['image_02/data','image_03/data']
    depth_list = ['proj_depth/groundtruth/image_02']
    train_list = search_dataset_folders(train_folders, num_images=num_images, num_targets=num_targets, cam_list=cam_list, extra_images_offset=5, depth_list=depth_list)
    test_list = search_dataset_folders(test_folders, num_images=num_images, num_targets=num_targets, cam_list=cam_list, extra_images_offset=5, depth_list=depth_list)
    return train_list, test_list


def kitti_depth_factory(root, split=None, transforms=None, loader=None, num_images=None, num_targets=None):
    train_list, test_list = make_dataset_splits(root, split, num_images=num_images, num_targets=num_targets)
    train_dataset = ListDataset(root, train_list, transforms[0], loader=loader)
    test_dataset = ListDataset(root, test_list, transforms[1], loader=loader)
    return train_dataset, test_dataset

def kitti_depth_factory_with_additional_info(root, split=None, transforms=None, loader=None, num_images=None, num_targets=None):
    train_list, test_list = make_dataset_splits(root, split, num_images=num_images, num_targets=num_targets)
    train_dataset = ListDatasetWithAdditionalInfo(root, train_list, transforms[0], loader=loader)
    test_dataset = ListDatasetWithAdditionalInfo(root, test_list, transforms[1], loader=loader)
    return train_dataset, test_dataset

def kitti_depth_loader(root, path_imgs, path_depths, additional_info=False):
    depth_imgs = [load_depth_from_png(depth) for depth in path_depths]
    imgs = [cv2.imread(img)[:,:,::-1] for img in path_imgs]
    imgs = [img.astype(np.uint8) for img in imgs]
    if additional_info:
        return imgs, depth_imgs, path_imgs, path_depths
    else:
        return imgs, depth_imgs


# sceneflow dataset is not avaliable - use blank flow target + depth for now
def kitti_depth_sceneflow_loader(root, path_imgs, path_depths, additional_info=False):
    #load depth
    depth_img = load_depth_from_png(path_depths[0])
    flow_zeros = np.zeros(depth_img.shape, dtype=np.float32)
    flow_img = np.stack((flow_zeros,flow_zeros), axis=2)
    sceneflow_imgs = (flow_img,depth_img)

    #load images
    imgs = [cv2.imread(img)[:,:,::-1] for img in path_imgs]
    imgs = [img.astype(np.uint8) for img in imgs]
    #
    if additional_info:
        return imgs, sceneflow_imgs, path_imgs, path_depths
    else:
        return imgs, sceneflow_imgs
