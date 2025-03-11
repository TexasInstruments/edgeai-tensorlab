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

# Dataset Location: https://www.kaggle.com/andrewmvd/tomato-detection, https://makeml.app/datasets/tomato
# About this dataset: This dataset contains 895 images with bounding box annotations provided in PASCAL VOC format for the creation of detection models.
# All annotations belong to a single class: tomato.
# Contributor: https://www.kaggle.com/andrewmvd


import os
import shutil
import collections
import json
import datetime

from .... import utils
from . import dataset_utils


tomatodet_categories = [dict(id=1, supercategory='tomato', name='tomato')]


def get_categories(state, example_project_path):
    return tomatodet_categories


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
        Tomato Dataset 
        * Type: Object Detection. The dataset is labeled with bounding box annotations. 
        * The dataset includes images of the following class - Tomato. 
        * URLs: https://www.kaggle.com/andrewmvd/tomato-detection, https://makeml.app/datasets/tomato
        * Contributor: https://www.kaggle.com/andrewmvd 
        ''')

    extract_root = os.path.join(example_project_path, 'other', 'extract')
    if (not force_download) and os.path.exists(extract_root) and \
        os.path.exists(example_project_files_path) and os.path.exists(example_project_annotations_path) and \
        os.path.exists(example_project_annotation_file_name):
        print(f'Dataset exists - will reuse: {example_project_path}')
        return (example_project_files_path,example_project_annotation_file_name)
    #

    dataset_urls = ['https://arcraftimages.s3-accelerate.amazonaws.com/Datasets/Tomato/TomatoPascalVOC.zip']

    download_root = os.path.join(example_project_path, 'other', 'download')
    utils.download_files(dataset_urls, download_root=download_root, extract_root=extract_root, log_writer=log_writer, progressbar_creator=progressbar_creator)

    # prepare the merge
    os.makedirs(example_project_files_path, exist_ok=True)
    os.makedirs(example_project_annotations_path, exist_ok=True)

    categories_list = get_categories(state, example_project_path)
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")

    dataset_store = dict()
    dataset_store['info'] = {
        'description': 'Tomato Detection Dataset',
        'version': '0.1',
        'year': '2020',
        'contributor': 'https://www.kaggle.com/andrewmvd',
        'data_created': date,
        'url': 'https://www.kaggle.com/andrewmvd/tomato-detection, https://makeml.app/datasets/tomato',
        'format': 'COCO 2017, https://cocodataset.org'
    }
    dataset_store['categories'] = tomatodet_categories
    dataset_store['licenses'] = None
    dataset_store['images'] = []
    dataset_store['annotations'] = []

    image_paths = ['images']
    anno_paths = ['annotations']
    image_id = 0
    for imageset_id, (image_path, anno_path) in enumerate(zip(image_paths, anno_paths)):
        image_path = os.path.join(extract_root, image_path)
        anno_path = os.path.join(extract_root, anno_path)
        for image_file in os.listdir(image_path):
            # copy image file
            image_file = image_file.rstrip()
            image_base = os.path.splitext(image_file)[0]
            src_file = os.path.join(image_path, image_base)+'.png'
            dst_file = os.path.join(example_project_files_path, image_base)+'.png'
            shutil.copy2(src_file, dst_file)
            # parse annotation file
            anno_file = os.path.join(anno_path, image_base)+'.xml'
            voc_dict = dataset_utils.parse_voc_xml_file(anno_file)

            file_name_partial = dataset_utils.get_file_name_partial(dst_file, state.projects_path)
            image_size = voc_dict['annotation']['size']
            image_info = {
                'id': image_id,
                'license': None,
                'file_name': os.path.basename(file_name_partial),
                'height': int(image_size['height']),
                'width': int(image_size['width']),
                'split_name': None
            }

            for object_anno_id, object_anno in enumerate(voc_dict['annotation']['object']):
                category_name = object_anno['name']
                annotation_value = object_anno['bndbox']
                annotation_bbox = {
                    'x': int(annotation_value['xmin']),
                    'y': int(annotation_value['ymin']),
                    'width': (int(annotation_value['xmax']) - int(annotation_value['xmin'])),
                    'height': (int(annotation_value['ymax']) - int(annotation_value['ymin']))
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
