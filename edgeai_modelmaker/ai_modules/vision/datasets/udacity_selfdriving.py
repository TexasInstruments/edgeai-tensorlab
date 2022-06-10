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


# Udacity Self Driving Car / Annotated Driving Dataset
# https://github.com/udacity/self-driving-car, https://github.com/udacity/self-driving-car/blob/master/LICENSE.md
# The dataset was annotated entirely by humans using Autti and is slightly larger with 15,000 frames.
# Labels: car, truck, pedestrian, biker, trafficLight


import os
import shutil
import collections
import json
import datetime
import PIL

from .... import utils
from . import dataset_utils


udacityselfdrive_categories = [dict(id=1, supercategory='car', name='car'),
                               dict(id=2, supercategory='truck', name='truck'),
                               dict(id=3, supercategory='pedestrian', name='pedestrian'),
                               dict(id=4, supercategory='biker', name='biker'),
                               dict(id=5, supercategory='trafficLight', name='trafficLight')]


def get_categories(state, example_project_path):
    return udacityselfdrive_categories


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

    
def dataset_download(state, example_project_path, force_download=False, log_writer=None, progressbar_creator=None):
    max_annotations_per_image = 1000
    example_dataset_path = os.path.join(example_project_path, 'dataset')
    example_project_files_path = os.path.join(example_dataset_path, 'images')
    example_project_annotations_path = os.path.join(example_dataset_path, 'annotations')
    example_project_annotation_file_name = os.path.join(example_project_annotations_path, 'instances.json')

    print(f'''
        Downloading and preparing dataset, this may take some time. 
        Udacity Self Driving Car / Annotated Driving Dataset 
        * URL: https://github.com/udacity/self-driving-car 
        * The dataset was annotated entirely by humans using Autti and is slightly larger with 15,000 frames. 
        * Labels: Car, Truck, Pedestrian, Biker, Street Lights 
        ''')

    extract_root = os.path.join(example_project_path, 'other', 'extract')
    if (not force_download) and os.path.exists(extract_root) and \
        os.path.exists(example_project_files_path) and os.path.exists(example_project_annotations_path) and \
        os.path.exists(example_project_annotation_file_name):
        print(f'dataset exists - will reuse: {example_project_path}')
        return (example_project_files_path,example_project_annotation_file_name)
    #

    dataset_urls = ['http://bit.ly/udacity-annotations-autti']
    save_filenames = ['object-dataset.tar.gz']

    download_root = os.path.join(example_project_path, 'other', 'download')
    utils.download_files(dataset_urls, download_root=download_root, extract_root=extract_root, save_filenames=save_filenames,
                         log_writer=log_writer, progressbar_creator=progressbar_creator)

    # prepare the merge
    os.makedirs(example_project_files_path, exist_ok=True)
    os.makedirs(example_project_annotations_path, exist_ok=True)

    categories_list = get_categories(state, example_project_path)
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")

    dataset_store = dict()
    dataset_store['info'] = {
        'description': 'Udacity Self Driving / Annotated Driving Dataset',
        'version': '0.1',
        'year': '2020',
        'contributor': 'Udacity',
        'data_created': date,
        'url': 'https://github.com/udacity/self-driving-car',
        'format': 'COCO 2017, https://cocodataset.org'
    }
    dataset_store['categories'] = categories_list
    dataset_store['licenses'] = None
    dataset_store['images'] = []
    dataset_store['annotations'] = []

    image_paths = ['object-dataset']
    anno_path = 'object-dataset/labels.csv'
    annotations_dict = dict()
    with open(os.path.join(extract_root, anno_path)) as fp:
        anno_lines = fp.readlines()
        for anno_line in anno_lines:
            anno_words = anno_line.rstrip().split(' ')
            anno_words = [anno_word.replace('"', '') for anno_word in anno_words]
            image_file = anno_words[0]
            image_file_base = os.path.splitext(image_file)[0]
            if image_file_base not in annotations_dict:
                annotations_dict[image_file_base] = []
            #
            annotations_dict[image_file_base].append(anno_words)
        #
    #

    image_id = 0
    for imageset_id, image_path in enumerate(image_paths):
        image_path = os.path.join(extract_root, image_path)
        anno_path = os.path.join(extract_root, anno_path)
        image_files = [f for f in os.listdir(image_path) if f.endswith('.jpg')]
        for image_file in image_files:
            # copy image file
            image_file = image_file.rstrip()
            image_base = os.path.splitext(image_file)[0]
            src_file = os.path.join(image_path, image_base)+'.jpg'
            dst_file = os.path.join(example_project_files_path, image_base)+'.jpg'
            shutil.copy2(src_file, dst_file)

            # parse annotation
            anno_list = annotations_dict[image_base] if image_base in annotations_dict else []

            file_name_partial = dataset_utils.get_file_name_partial(dst_file, state.projects_path)
            image_size = PIL.Image.open(src_file).size
            image_info = {
                'id': image_id,
                'license': None,
                'file_name': os.path.basename(file_name_partial),
                'height': image_size[1],
                'width': image_size[0],
                'split_name': None
            }

            for object_anno_id, object_anno in enumerate(anno_list):
                category_name = object_anno[6]
                annotation_bbox = {
                    'x': int(object_anno[1]),
                    'y': int(object_anno[2]),
                    'width': (int(object_anno[3]) - int(object_anno[1])),
                    'height': (int(object_anno[4]) - int(object_anno[2]))
                }
                annotation_info = {
                    'id': image_id * max_annotations_per_image + object_anno_id,
                    'segmentation': None,
                    'area': None,
                    'iscrowd': 0,
                    'image_id': image_id,
                    'category_id': dataset_utils.get_category_id(categories_list, category_name),
                    'bbox': list(annotation_bbox.values())
                }
                dataset_store['annotations'].append(annotation_info)
            #
            dataset_store['images'].append(image_info)
            image_id = image_id + 1
        #
    #

    with open(example_project_annotation_file_name, 'w') as fp:
        json.dump(dataset_store, fp)
    #
    return (example_project_files_path,example_project_annotation_file_name)
