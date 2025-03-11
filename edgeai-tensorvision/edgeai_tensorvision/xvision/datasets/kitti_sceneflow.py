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
from .dataset_utils import split2list, ListDataset
import cv2
import numpy as np
from edgeai_torchmodelopt import xnn

__all__ = ['kitti_flow_occ','kitti_flow_noc','kitti_sceneflow_occ']

'''
Dataset routines for kitti_flow, 2012 and 2015.
http://www.cvlibs.net/datasets/kitti/eval_flow.php
The dataset is not very big, you might want to only finetune on it for flownet
EPE are not representative in this dataset because of the sparsity of the GT.
'''

def kitti_flow_occ(dataset_config, root, transforms=None, split=80):
    train_list, test_list = make_dataset(root, split, True)
    train_dataset = ListDataset(root, train_list, transforms[0], loader=kitti_flow_loader)
    # All test sample are cropped to lowest possible size of KITTI images
    test_dataset = ListDataset(root, test_list, transforms[1], loader=kitti_flow_loader)
    return train_dataset, test_dataset


def kitti_flow_noc(dataset_config, root, transforms=None, split=80):
    train_list, test_list = make_dataset(root, split, False)
    train_dataset = ListDataset(root, train_list, transforms[0], loader=kitti_flow_loader)
    # All test sample are cropped to lowest possible size of KITTI images
    test_dataset = ListDataset(root, test_list, transforms[1], loader=kitti_flow_loader)
    return train_dataset, test_dataset


def kitti_sceneflow_occ(dataset_config, root, transforms=None, split=80):
    print('{}=> kitti_occ_sceneflow dataset has empty depth. Sparse estimation mode may be needed to ignore it {}'
          .format(xnn.utils.TermColor.YELLOW, xnn.utils.TermColor.END))
    train_list, test_list = make_dataset(root, split, True)
    train_dataset = ListDataset(root, train_list, transforms[0], loader=kitti_occ_sceneflow_loader)
    # All test sample are cropped to lowest possible size of KITTI images
    test_dataset = ListDataset(root, test_list, transforms[1], loader=kitti_occ_sceneflow_loader)
    return train_dataset, test_dataset



############################################################################
# internal functions
############################################################################

def load_flow_from_png(png_path):
    flo_file = cv2.imread(png_path,cv2.IMREAD_UNCHANGED)
    flo_img = flo_file[:,:,2:0:-1].astype(np.float32)
    invalid = (flo_file[:,:,0] == 0)
    flo_img = flo_img - 32768
    flo_img = flo_img / 64
    flo_img[np.abs(flo_img) < 1e-10] = 1e-10
    flo_img[invalid, :] = 0
    return(flo_img)


def make_dataset(dir, split, occ=True):
    '''Will search in training folder for folders 'flow_noc' or 'flow_occ' and 'colored_0' (KITTI 2012) or 'image_2' (KITTI 2015) '''
    flow_dir = 'flow_occ' if occ else 'flow_noc'
    assert(os.path.isdir(os.path.join(dir,flow_dir)))
    img_dir = 'colored_0'
    if not os.path.isdir(os.path.join(dir,img_dir)):
        img_dir = 'image_2'
    assert(os.path.isdir(os.path.join(dir,img_dir)))

    images = []
    for flow_map in glob.iglob(os.path.join(dir,flow_dir,'*.png')):
        flow_map = os.path.basename(flow_map)
        root_filename = flow_map[:-7]
        flow_map = os.path.join(flow_dir,flow_map)
        img1 = os.path.join(img_dir,root_filename+'_10.png')
        img2 = os.path.join(img_dir,root_filename+'_11.png')
        if not (os.path.isfile(os.path.join(dir,img1)) or os.path.isfile(os.path.join(dir,img2))):
            continue
        images.append([[img1,img2],flow_map])
    #
    images.sort()
    return split2list(images, split)


def kitti_flow_loader(root,path_imgs, path_flo):
    imgs = [os.path.join(root,path) for path in path_imgs]
    imgs = [cv2.imread(img)[:,:,::-1].astype(np.float32) for img in imgs]
    flo = os.path.join(root,path_flo)
    flo = [load_flow_from_png(flo)]
    return imgs,flo


def kitti_occ_sceneflow_loader(root,path_imgs, path_flo):
    imgs = [os.path.join(root,path) for path in path_imgs]
    flo = os.path.join(root,path_flo)
    imgs = [cv2.imread(img)[:,:,::-1].astype(np.float32) for img in imgs]
    flo=load_flow_from_png(flo)
    flow_zeros = np.zeros((flo.shape[0],flo.shape[1],1), dtype=np.float)
    flo = [flow_zeros,flo]
    return imgs,flo