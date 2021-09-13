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
Reference: http://sintel.is.tue.mpg.de/

Butler, D. J. and Wulff, J. and Stanley, G. B. and Black, M. J.,
A naturalistic open source movie for optical flow evaluation,
European Conf. on Computer Vision (ECCV), 2012

Lessons and insights from creating a synthetic optical flow benchmark,
Wulff, J. and Butler, D. J. and Stanley, G. B. and Black, M. J.,
ECCV Workshop on Unsolved Problems in Optical Flow and Stereo Estimation,
2012
"""

import os.path
import glob
from .dataset_utils import split2list, ListDataset

# Requirements: Numpy as PIL/Pillow
import numpy as np
import cv2

__all__ = ['mpi_sintel_clean','mpi_sintel_final','mpi_sintel_both', 'mpi_sintel_depth', 'mpi_sintel_sceneflow']

# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'
MAX_DEPTH = 150

'''
Dataset routines for MPI Sintel.
http://sintel.is.tue.mpg.de/
clean version imgs are without shaders, final version imgs are fully rendered
The dataset is not very big, you might want to only pretrain on it for flownet
'''
############################################################################
def mpi_sintel_clean(dataset_config, root, split=None, transforms=None):
    train_list, test_list = make_dataset_flow(root, split, 'clean')
    train_dataset = ListDataset(root, train_list, transforms[0])
    test_dataset = ListDataset(root, test_list, transforms[1])
    return train_dataset, test_dataset


def mpi_sintel_final(dataset_config, root, split=None, transforms=None):
    train_list, test_list = make_dataset_flow(root, split, 'final')
    train_dataset = ListDataset(root, train_list, transforms[0])
    test_dataset = ListDataset(root, test_list, transforms[1])
    return train_dataset, test_dataset


def mpi_sintel_both(dataset_config, root, split=None, transforms=None):
    '''load images from both clean and final folders.
    We cannot shuffle input, because it would very likely cause data snooping
    for the clean and final frames are not that different'''
    train_list1, test_list1 = make_dataset_flow(root, split, 'clean')
    train_list2, test_list2 = make_dataset_flow(root, split, 'final')
    train_dataset = ListDataset(root, train_list1 + train_list2, transforms[0])
    test_dataset = ListDataset(root, test_list1 + test_list2, transforms[1])
    return train_dataset, test_dataset


############################################################################
def mpi_sintel_depth(dataset_config, root, split=None, transforms=None):
    train_list, test_list = make_dataset_depth(root, split, 'depth', num_target=1)
    train_dataset = ListDataset(root, train_list, transforms[0], loader=mpi_sintel_depth_loader1)
    test_dataset = ListDataset(root, test_list, transforms[1], loader=mpi_sintel_depth_loader1)
    return train_dataset, test_dataset


def mpi_sintel_sceneflow(dataset_config, root, split=None, transforms=None):
    train_list, test_list = make_dataset_depth(root, split, 'sceneflow', num_target=3)
    train_dataset = ListDataset(root, train_list, transforms[0], loader=mpi_sintel_sceneflow_loader)
    test_dataset = ListDataset(root, test_list, transforms[1], loader=mpi_sintel_sceneflow_loader)
    return train_dataset, test_dataset


############################################################################
# internal functions
############################################################################
def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))
    return data2D


def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    depth = np.minimum(depth,MAX_DEPTH)
    return depth



def make_dataset_flow(dir, split, dataset_type='clean'):
    flow_dir = 'flow'
    img_dir = dataset_type
    images = []
    for flow_map in glob.iglob(os.path.join(dir,flow_dir,'*','*.flo')):
        flow_map = os.path.relpath(flow_map,os.path.join(dir,flow_dir))
        root_filename = flow_map[:-8]
        frame_nb = int(flow_map[-8:-4])
        img1 = os.path.join(img_dir,root_filename+str(frame_nb).zfill(4)+'.png')
        img2 = os.path.join(img_dir,root_filename+str(frame_nb+1).zfill(4)+'.png')
        flow_map = os.path.join(flow_dir,flow_map)
        if not (os.path.isfile(os.path.join(dir,img1)) or os.path.isfile(os.path.join(dir,img2))):
            continue

        imgs = [img1,img2]
        flow_maps = [flow_map]
        images.append([imgs,flow_maps])

    return split2list(images, split)


######################################################################################

def make_dataset_depth(dir, split, dataset_type='depth', num_target=None):
    depth_dir = 'depth'
    img_dir = 'final'
    images = []
    folder_list = sorted(glob.glob(os.path.join(dir,depth_dir,'*')))
    for folder in folder_list:
        depth_list = sorted(glob.glob(os.path.join(dir,depth_dir,folder,'*.dpt')))
        for depth_map1 in depth_list[:-1]:
            depth_map1 = os.path.relpath(depth_map1,os.path.join(dir,depth_dir))
            root_filename = depth_map1[:-8]
            frame_nb = int(depth_map1[-8:-4])
            img1 = os.path.join(img_dir,root_filename+str(frame_nb).zfill(4)+'.png')
            img2 = os.path.join(img_dir,root_filename+str(frame_nb+1).zfill(4)+'.png')
            if not (os.path.isfile(os.path.join(dir,img1)) or os.path.isfile(os.path.join(dir,img2))):
                continue

            depth_map1 = os.path.join(depth_dir, depth_map1)
            if num_target == 2:
                depth_map2 = os.path.join(depth_dir, root_filename + str(frame_nb + 1).zfill(4) + '.dpt')
                images.append([[img1,img2],[depth_map1,depth_map2]])
            else:
                images.append([[img1,img2],[depth_map1]])

    return split2list(images, split, default_split=0.87)

# target depth for only the first image
def mpi_sintel_depth_loader1(root, path_imgs, path_depths):
    path_depths = [os.path.join(root,path) for path in path_depths]
    depth_img = [depth_read(path_depths[0])]
    imgs = [os.path.join(root,path) for path in path_imgs]
    imgs = [cv2.imread(img) for img in imgs]
    imgs = [img[:,:,::-1] for img in imgs]
    imgs = [img.astype(np.float32) for img in imgs]
    return imgs,depth_img


# target sceneflow (depth is for only the first image)
def mpi_sintel_sceneflow_loader(root, path_imgs, path_target):
    path_target = [os.path.join(root,path) for path in path_target]
    flow_img = load_flo(path_target[0])
    depth_img = depth_read(path_target[1])[...,np.newaxis]
    imgs = [os.path.join(root,path) for path in path_imgs]

    imgs = [cv2.imread(img) for img in imgs]
    imgs = [img[:,:,::-1] for img in imgs]
    imgs = [img.astype(np.float32) for img in imgs]

    target_imgs = (flow_img,depth_img)
    return imgs,target_imgs
