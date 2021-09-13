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

import os
import numpy as np
import torch.utils
import cv2


def split2list(images, split):
    if isinstance(split, str):
        with open(split) as f:
            split_values = [x.strip() == '1' for x in f.readlines()]
        assert(len(images) == len(split_values))
    elif isinstance(split, float):
        split_values = np.random.uniform(0,1,len(images)) < split
    else:
        assert False, 'split could not be understood'
    #
    train_samples = [sample for sample, sval in zip(images, split_values) if sval]
    test_samples = [sample for sample, sval in zip(images, split_values) if not sval]
    return train_samples, test_samples


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


def default_loader(root, path_imgs, path_flows):
    imgs = [os.path.join(root,path) for path in path_imgs]
    flows = [os.path.join(root,path_flo) for path_flo in path_flows]
    imgs = [cv2.imread(img)[:,:,::-1] for img in imgs]
    imgs = [img.astype(np.float32) for img in imgs]
    flows = [load_flo(flo) for flo in flows]
    return imgs,flows


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, root, path_list, transform=None, loader=default_loader):
        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        inputs, targets = self.path_list[index]
        inputs, targets = self.loader(self.root, inputs, targets)
        if self.transform is not None:
            inputs, targets = self.transform(inputs, targets)
        #
        return inputs, targets

    def __len__(self):
        return len(self.path_list)


class ListDatasetWithAdditionalInfo(torch.utils.data.Dataset):
    def __init__(self, root, path_list, transform=None, loader=default_loader):
        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        inputs, targets = self.path_list[index]
        inputs, targets, input_paths, target_paths = self.loader(self.root, inputs, targets, additional_info=True)
        if self.transform is not None:
            inputs, targets = self.transform(inputs, targets)
        return inputs, targets, input_paths, target_paths

    def __len__(self):
        return len(self.path_list)