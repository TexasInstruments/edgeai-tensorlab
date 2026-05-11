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
import json
import os
import random

import numpy as np
import torch.utils
import cv2


def split2list(images, split):
    if isinstance(split, str):
        with open(split) as f:
            split_values = [x.strip() == '1' for x in f.readlines()]
        assert (len(images) == len(split_values))
    elif isinstance(split, float):
        split_values = np.random.uniform(0, 1, len(images)) < split
    else:
        assert False, 'split could not be understood'
    #
    train_samples = [sample for sample, sval in zip(images, split_values) if sval]
    test_samples = [sample for sample, sval in zip(images, split_values) if not sval]
    return train_samples, test_samples


def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))
    return data2D


def _find_annotations_info(dataset_store):
    image_id_to_file_id_dict = dict()
    file_id_to_image_id_dict = dict()
    annotations_info_list = []
    for file_id, image_info in enumerate(dataset_store['images']):
        image_id = image_info['id']
        image_id_to_file_id_dict[image_id] = file_id
        file_id_to_image_id_dict[file_id] = image_id
        annotations_info_list.append([])
    #
    for annotation_info in dataset_store['annotations']:
        if annotation_info:
            image_id = annotation_info['image_id']
            file_id = image_id_to_file_id_dict[image_id]
            annotations_info_list[file_id].append(annotation_info)
        #
    #
    return annotations_info_list


def dataset_split(dataset, split_factor, split_names=('train', 'val'), random_seed=1):
    random.seed(random_seed)
    if isinstance(dataset, str):
        with open(dataset) as fp:
            dataset = json.load(fp)
        #
    #

    dataset_train = dict(info=dataset['info'],
                         categories=dataset['categories'],
                         images=[], annotations=[])
    dataset_val = dict(info=dataset['info'],
                       categories=dataset['categories'],
                       images=[], annotations=[])
    dataset_splits = {split_names[0]: dataset_train, split_names[1]: dataset_val}

    annotations_info_list = _find_annotations_info(dataset)
    image_count_split = {split_name: 0 for split_name in split_names}
    for image_id, (image_info, annotations) in enumerate(zip(dataset['images'], annotations_info_list)):
        if not annotations:
            # ignore images without annotations from the splits
            continue
        #
        image_info['file_name'] = os.path.basename(image_info['file_name'])
        if 'split_name' in image_info and image_info['split_name'] is not None:
            # print(f'file_name={image_info["file_name"]} split_name={image_info["split_name"]}')
            split_name = image_info['split_name']
            split_name = split_names[0] if 'train' in split_name else split_name  # change trainval to tarin
            split_name = split_names[1] if 'test' in split_name else split_name  # change test to val
        else:
            # print(f'split_name was not found in {image_info["file_name"]}')
            split_name = split_names[0] if random.random() < split_factor else split_names[1]
        #
        dataset_splits[split_name]['images'].append(image_info)
        dataset_splits[split_name]['annotations'].extend(annotations)
        image_count_split[split_name] += 1
    #
    print('dataset split sizes', image_count_split)
    return dataset_splits


def default_loader(root, path_imgs, path_flows):
    imgs = [os.path.join(root, path) for path in path_imgs]
    flows = [os.path.join(root, path_flo) for path_flo in path_flows]
    imgs = [cv2.imread(img)[:, :, ::-1] for img in imgs]
    imgs = [img.astype(np.float32) for img in imgs]
    flows = [load_flo(flo) for flo in flows]
    return imgs, flows


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
