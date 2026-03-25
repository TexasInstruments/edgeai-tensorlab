# Copyright (c) 2018-2021, Texas Instruments Incorporated
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

import json
import os
import sys
import warnings
import datetime
import random
import itertools
import PIL

from .... import utils
from . import dataset_utils
from . import widerface_detection
from . import pascal_voc0712
from . import udacity_selfdriving
from . import coco_detection
from . import tomato_detection
from . import oxford_flowers102


def get_datasets_list(task_type=None):
    if task_type == 'detection':
        return ['widerface_detection', 'pascal_voc0712', 'coco_detection', 'udacity_selfdriving', 'tomato_detection']
    elif task_type == 'classification':
        return ['oxford_flowers102']
    else:
        assert False, 'unknown task type for get_datasets_list'


def get_target_module(backend_name):
    this_module = sys.modules[__name__]
    target_module = getattr(this_module, backend_name)
    return target_module


class DatasetHandling:
    @classmethod
    def init_params(self, *args, **kwargs):
        params = dict(
            dataset=dict(
            )
        )
        params = utils.ConfigDict(params, *args, **kwargs)
        return params

    def __init__(self, *args, quit_event=None, **kwargs):
        self.params = self.init_params(*args, **kwargs)
        self.quit_event = quit_event
        self._gen_dataset_split_paths()

    def _gen_dataset_split_paths(self):
        self.params.dataset.data_path_splits = []
        self.params.dataset.annotation_path_splits = []
        for split_idx, split_name in enumerate(self.params.dataset.split_names):
            self.params.dataset.data_path_splits.append(os.path.join(self.params.dataset.dataset_path, split_name))
            annotation_path_split = os.path.join(self.params.dataset.dataset_path, self.params.dataset.annotation_dir,
                             f'{self.params.dataset.annotation_prefix}_{split_name}.json')
            self.params.dataset.annotation_path_splits.append(annotation_path_split)
        #

    def clear(self):
        pass

    def run(self):
        max_num_files = self.get_max_num_fies()
        # dataset reading/splitting
        if isinstance(self.params.dataset.input_data_path, (list, tuple)) and \
                isinstance(self.params.dataset.input_annotation_path, (list, tuple)):
            # dataset splits are directly given
            dataset_splits = dict()
            for split_idx, split_name in enumerate(self.params.dataset.split_names):
                dataset_splits[split_name] = dataset_utils.dataset_load(
                    self.params.dataset.input_annotation_path[split_idx],
                    self.params.dataset.input_data_path[split_idx],
                    self.params.common.task_type,
                    annotation_format=self.params.dataset.annotation_format)
                if dataset_splits[split_name] is None:
                    print('ERROR: could not load dataset')
                    return False
                #
                dataset_splits[split_name] = dataset_utils.dataset_split_limit(dataset_splits[split_name],
                                                                               max_num_files[split_idx])
                dataset_utils.dataset_split_write(
                    self.params.dataset.input_data_path[split_idx], dataset_splits[split_name],
                    self.params.dataset.data_path_splits[split_idx],
                    self.params.dataset.annotation_path_splits[split_idx])
                dataset_utils.dataset_split_link(
                    self.params.dataset.input_data_path[split_idx], dataset_splits[split_name],
                    self.params.dataset.data_path_splits[split_idx], self.params.dataset.annotation_path_splits[split_idx])
            #
        elif self.params.dataset.input_data_path is not None and isinstance(self.params.dataset.input_annotation_path, (list, tuple)):
            dataset_splits = dict()
            # Make the locally available dataset symlinks
            download_root = os.path.join(self.params.common.download_path, 'datasets')
            _, _, input_data_path = utils.download_file(self.params.dataset.input_data_path, download_root, self.params.dataset.extract_path)
            self.params.dataset.input_data_path = input_data_path
            # Loading datasets splits
            for split_idx, split_name in enumerate(self.params.dataset.split_names):
                dataset_splits[split_name] = dataset_utils.dataset_load(
                    self.params.dataset.input_annotation_path[split_idx],
                    self.params.dataset.input_data_path[split_idx],
                    self.params.common.task_type,
                    annotation_format=self.params.dataset.annotation_format)
                if dataset_splits[split_name] is None:
                    print('ERROR: could not load dataset')
                    return False
                #
            #
        else:
            if isinstance(self.params.dataset.input_data_path, str) and isinstance(self.params.dataset.input_annotation_path, str):
                input_annotation_path = self.params.dataset.input_annotation_path
                download_root = os.path.join(self.params.common.download_path, 'datasets')
                _, _, input_data_path = utils.download_file(self.params.dataset.input_data_path, download_root, self.params.dataset.extract_path)
                self.params.dataset.input_data_path, self.params.dataset.input_annotation_path = input_data_path, input_annotation_path
                self.params.dataset.annotation_prefix = os.path.splitext(os.path.basename(input_annotation_path))[0]
                self._gen_dataset_split_paths()
            elif isinstance(self.params.dataset.input_data_path, str):
                input_annotation_path = os.path.join(self.params.dataset.dataset_path, self.params.dataset.annotation_dir, f'{self.params.dataset.annotation_prefix}.json')
                download_root = os.path.join(self.params.common.download_path, 'datasets')
                _, _, input_data_path = utils.download_file(self.params.dataset.input_data_path, download_root, self.params.dataset.extract_path)
                self.params.dataset.input_data_path, self.params.dataset.input_annotation_path = input_data_path, input_annotation_path
            elif self.params.dataset.dataset_name in get_datasets_list(self.params.common.task_type):
                dataset_backend = get_target_module(self.params.dataset.dataset_name)
                if self.params.dataset.dataset_reload:
                    dataset_download_paths = dataset_backend.dataset_reload(self.params, self.params.common.project_path)
                elif self.params.dataset.dataset_download:
                    dataset_download_paths = dataset_backend.dataset_download(self.params, self.params.common.project_path)
                else:
                    dataset_download_paths = dataset_backend.dataset_paths(self.params, self.params.common.project_path)
                #
                input_images_path, self.params.dataset.input_annotation_path = dataset_download_paths
                self.params.dataset.input_data_path = input_images_path.replace(self.params.dataset.data_dir, '')
            else:
                print('ERROR: invalid dataset details')
                return False
            #

            # data (images) folder and annotation folder are given
            dataset_store = dataset_utils.dataset_load(
                self.params.dataset.input_annotation_path,
                self.params.dataset.input_data_path,
                self.params.common.task_type,
                annotation_format=self.params.dataset.annotation_format)
            if dataset_store is None:
                print('ERROR: could not load dataset')
                return False
            #

            # split the dataset into train/val
            dataset_splits = dataset_utils.dataset_split(dataset_store,
                                         self.params.dataset.split_factor,
                                         self.params.dataset.split_names)

            # write dataset splits
            for split_idx, split_name in enumerate(dataset_splits):
                input_images_path = os.path.join(self.params.dataset.input_data_path, self.params.dataset.data_dir)
                dataset_splits[split_name] = dataset_utils.dataset_split_limit(dataset_splits[split_name], max_num_files[split_idx])
                dataset_utils.dataset_split_write(
                    input_images_path, dataset_splits[split_name],
                    self.params.dataset.data_path_splits[split_idx],
                    self.params.dataset.annotation_path_splits[split_idx])
                dataset_utils.dataset_split_link(
                    input_images_path, dataset_splits[split_name],
                    self.params.dataset.data_path_splits[split_idx],
                    self.params.dataset.annotation_path_splits[split_idx])
            #
        #
        if self.params.dataset.max_num_files is not None:
            print(f'INFO: ModelMaker - max_num_files is set to: {self.params.dataset.max_num_files}')
            print(f'INFO: ModelMaker - dataset split sizes are limited to:', \
                  {split_name: len(dataset_split['images']) for split_name, dataset_split in dataset_splits.items()})
        #
        print('INFO: ModelMaker - dataset loading OK')
        return True

    def get_max_num_fies(self):
        if isinstance(self.params.dataset.max_num_files, (list, tuple)):
            max_num_files = self.params.dataset.max_num_files
        elif isinstance(self.params.dataset.max_num_files, int):
            assert (0.0 < self.params.dataset.split_factor < 1.0), 'split_factor must be between 0 and 1.0'
            assert len(self.params.dataset.split_names) > 1, 'split_names must have at least two entries'
            max_num_files = [None] * len(self.params.dataset.split_names)
            for split_id, split_name in enumerate(self.params.dataset.split_names):
                if split_id == 0:
                    max_num_files[split_id] = round(self.params.dataset.max_num_files * self.params.dataset.split_factor)
                else:
                    max_num_files[split_id] = self.params.dataset.max_num_files - max_num_files[0]
                #
            #
        else:
            warnings.warn('unrecognized value for max_num_files - must be int, list or tuple')
            assert len(self.params.dataset.split_names) > 1, 'split_names must have at least two entries'
            max_num_files = [None] * len(self.params.dataset.split_names)
        #
        return max_num_files

    def get_params(self):
        return self.params
