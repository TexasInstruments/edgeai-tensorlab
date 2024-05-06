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

# Dataset Location: http://host.robots.ox.ac.uk/pascal/VOC/
# The PASCAL Visual Object Classes (VOC) Challenge
# Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman, A.
# International Journal of Computer Vision, 88(2), 303-338, 2010


import json
import os
import datetime
import shutil

from .... import utils
from . import dataset_utils


def get_categories(state, example_project_path):
    example_dataset_path = os.path.join(example_project_path, 'dataset')
    example_project_annotations_path = os.path.join(example_dataset_path, 'annotations')
    example_project_annotation_file_name = os.path.join(example_project_annotations_path, 'instances.json')
    coco_dict = json.load(example_project_annotation_file_name)
    return coco_dict


def dataset_paths(state, example_project_path, **kwargs):
    example_dataset_path = os.path.join(example_project_path, 'dataset')
    example_project_files_path = os.path.join(example_dataset_path, 'images')
    example_project_annotations_path = os.path.join(example_dataset_path, 'annotations')
    example_project_annotation_file_name = os.path.join(example_project_annotations_path, 'instances.json')
    return (example_project_files_path,example_project_annotation_file_name)


def dataset_reload(state, example_project_path, force_download=False, log_writer=None,
                   progressbar_creator=None):
    example_dataset_path = os.path.join(example_project_path, 'dataset')
    shutil.rmtree(example_dataset_path, ignore_errors=True)
    return dataset_download(state, example_project_path, force_download, log_writer=log_writer,
                     progressbar_creator=progressbar_creator)

    
def dataset_download(state, example_project_path, force_download=False, log_writer=None,
                     progressbar_creator=None):
    example_dataset_path = os.path.join(example_project_path, 'dataset')
    example_project_files_path = os.path.join(example_dataset_path, 'images')
    example_project_annotations_path = os.path.join(example_dataset_path, 'annotations')
    example_project_annotation_file_name = os.path.join(example_project_annotations_path, 'instances.json')

    print(f'''
        Downloading and preparing dataset, this may take some time.
        COCO Dataset:
        * Microsoft COCO: Common Objects in Context, 
        *  Tsung-Yi Lin, et.al. https://arxiv.org/abs/1405.0312
        *  URL: https://cocodataset.org/ 
        ''')

    extract_root = os.path.join(example_project_path, 'other', 'extract')
    coco_train_images_path = os.path.join(extract_root, 'train2017')
    coco_val_images_path = os.path.join(extract_root, 'val2017')
    if (not force_download) and os.path.exists(extract_root) and \
        os.path.exists(coco_train_images_path) and os.path.exists(coco_val_images_path) and \
        os.path.exists(example_project_files_path) and os.path.exists(example_project_annotations_path) and \
        os.path.exists(example_project_annotation_file_name):
        print(f'Dataset exists - will reuse: {example_project_path}')
        return (example_project_files_path,example_project_annotation_file_name)
    #

    dataset_urls = ['http://images.cocodataset.org/zips/train2017.zip',
                    'http://images.cocodataset.org/zips/val2017.zip',
                    'http://images.cocodataset.org/annotations/annotations_trainval2017.zip']

    download_root = os.path.join(example_project_path, 'other', 'download')
    utils.download_files(dataset_urls, download_root=download_root, extract_root=extract_root, log_writer=log_writer,
                         progressbar_creator=progressbar_creator)

    # prepare the VOC0712 merge
    os.makedirs(example_project_files_path, exist_ok=True)
    os.makedirs(example_project_annotations_path, exist_ok=True)

    image_dir = os.path.join(extract_root, 'train2017')
    image_files = os.listdir(image_dir)
    for image_file in image_files:
        src_image = os.path.join(image_dir, image_file)
        dst_image = os.path.join(example_project_files_path, image_file)
        shutil.copy2(src_image, dst_image)
    #
    src_anno_file = os.path.join(extract_root, 'annotations', 'instances_train2017.json')
    shutil.copy2(src_anno_file, example_project_annotation_file_name)
    return (example_project_files_path,example_project_annotation_file_name)
