# Copyright (c) 2018-2022, Texas Instruments Incorporated
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
import shutil
import collections
import json
import datetime
import PIL

from . import dataset_utils


tiscapes2017_driving_categories = [
    dict(id=0, supercategory='background', name='background'),
    dict(id=0, supercategory='road', name='road'),
    dict(id=0, supercategory='human', name='human'),
    dict(id=0, supercategory='roadsign', name='roadsign'),
    dict(id=0, supercategory='vehicle', name='vehicle'),
]


def get_categories(state, example_project_path):
    return tiscapes2017_driving_categories


def dataset_paths(state, example_project_path, **kwargs):
    example_dataset_path = os.path.join(example_project_path, 'dataset')
    example_project_files_path = os.path.join(example_dataset_path, 'images')
    example_project_annotations_path = os.path.join(example_dataset_path, 'annotations')
    example_project_annotation_file_name = os.path.join(example_project_annotations_path, 'instances.json')
    return (example_project_files_path,example_project_annotation_file_name)


def dataset_reload(state, example_project_path, force_download=False, log_writer=None, progressbar_creator=None):
    example_dataset_path = os.path.join(example_project_path, 'dataset')
    shutil.rmtree(example_dataset_path, ignore_errors=True)
    return dataset_download(state, example_project_path, force_download, log_writer=log_writer, progressbar_creator=progressbar_creator)


def dataset_download(state, example_project_path, force_download=False, log_writer=None, progressbar_creator=None,
                     remove_category='background'):
    max_annotations_per_image = 1000
    example_dataset_path = os.path.join(example_project_path, 'dataset')
    example_project_files_path = os.path.join(example_dataset_path, 'images')
    example_project_annotations_path = os.path.join(example_dataset_path, 'annotations')
    example_project_annotation_file_name = os.path.join(example_project_annotations_path, 'instances.json')

    print(f'''
        Downloading and preparing dataset, this may take some time. 
        ''')

    # extract_root = os.path.join(example_project_path, 'other', 'extract')
    if (not force_download) and \
        os.path.exists(example_project_files_path) and os.path.exists(example_project_annotations_path) and \
        os.path.exists(example_project_annotation_file_name):
        print(f'Dataset exists - will reuse: {example_project_path}')
        return (example_project_files_path,example_project_annotation_file_name)
    #

    dataset_urls = ['./data/labelling/tiscapes2017_driving.zip']

    download_root = os.path.join(example_project_path, 'other', 'download')
    dataset_utils.download_files(dataset_urls, download_root=download_root, extract_root=example_dataset_path,
                                 log_writer=log_writer, progressbar_creator=progressbar_creator)

    # remove background class as it cannot be used for bbox training (it is also not completely annotated)
    with open(example_project_annotation_file_name) as fp:
        dataset_store = json.load(fp)
    #
    if dataset_store['categories'][0]['name'] == remove_category:
        dataset_store['categories'].pop(0)
    #
    annotations = []
    for anno_id, anno in enumerate(dataset_store['annotations']):
        if anno['category_id'] != 0:
            annotations.append(anno)
        #
    #
    dataset_store['annotations'] = annotations
    with open(example_project_annotation_file_name, 'w') as fp:
        json.dump(dataset_store, fp)
    #
    return (example_project_files_path,example_project_annotation_file_name)
