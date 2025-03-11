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

import os
import shutil
import collections
import json
import datetime

from .... import utils
from . import dataset_utils


def get_categories(state, example_project_path):
    example_extract_path = os.path.join(example_project_path, 'other', 'extract')
    path = os.path.join(example_extract_path, 'VOCdevkit')
    categories_path = os.path.join(path, 'VOC2007/ImageSets/Main')
    categories_list = [f.split('.')[0].split('_')[0] for f in os.listdir(categories_path) if f.endswith('_train.txt')]
    categories_list = list(set(categories_list))
    categories_dict = [dict(id=cat_id+1, supercategory=cat_name, name=cat_name) for cat_id, cat_name in enumerate(categories_list)]
    return categories_dict


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
        Pascal VOC 2012 Dataset (VOC2012), Pascal VOC 2007 Dataset (VOC2007): 
        * The PASCAL Visual Object Classes (VOC) Challenge, Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J. and Zisserman, A., 
        * International Journal of Computer Vision, 88(2), 303-338, 2010, 
        * URL: http://host.robots.ox.ac.uk/pascal/VOC/
        ''')

    extract_root = os.path.join(example_project_path, 'other', 'extract')
    voc20212_path = os.path.join(extract_root, 'VOCdevkit', 'VOC2012')
    voc2007_path = os.path.join(extract_root, 'VOCdevkit', 'VOC2007')
    if (not force_download) and os.path.exists(extract_root) and os.path.exists(voc2007_path) and os.path.exists(voc20212_path) and \
        os.path.exists(example_project_files_path) and os.path.exists(example_project_annotations_path) and \
        os.path.exists(example_project_annotation_file_name):
        print(f'Dataset exists - will reuse: {example_project_path}')
        return (example_project_files_path,example_project_annotation_file_name)
    #

    dataset_urls = ['http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
                    'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
                    'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
                    'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar']

    download_root = os.path.join(example_project_path, 'other', 'download')
    utils.download_files(dataset_urls, download_root=download_root, extract_root=extract_root, log_writer=log_writer, progressbar_creator=progressbar_creator)

    # prepare the VOC0712 merge
    os.makedirs(example_project_files_path, exist_ok=True)
    os.makedirs(example_project_annotations_path, exist_ok=True)

    categories_list = get_categories(state, example_project_path)
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")

    dataset_store = dict()
    dataset_store['info'] = {
        'description': 'Pascal VOC 2007 2012 Dataset',
        'version': '0.1',
        'year': '2007 2012',
        'contributor': 'Pascal VOC 2007, 2012 Dataset (VOC2007, VOC2012) The PASCAL Visual Object Classes (VOC) Challenge, Everingham, M., Van Gool et.al.',
        'data_created': date,
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/',
        'format': 'COCO 2017, https://cocodataset.org'
    }
    dataset_store['categories'] = categories_list
    dataset_store['licenses'] = None
    dataset_store['images'] = []
    dataset_store['annotations'] = []

    image_paths = ['VOCdevkit/VOC2012/JPEGImages', 'VOCdevkit/VOC2007/JPEGImages', 'VOCdevkit/VOC2007/JPEGImages']
    imageset_paths = ['VOCdevkit/VOC2012/ImageSets/Main/trainval.txt', 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'VOCdevkit/VOC2007/ImageSets/Main/test.txt']
    anno_paths = ['VOCdevkit/VOC2012/Annotations', 'VOCdevkit/VOC2007/Annotations', 'VOCdevkit/VOC2007/Annotations']
    split_names = ['trainval', 'trainval', 'test']
    image_id = 0
    for imageset_id, (image_path, imageset_path, anno_path, split_name) in \
            enumerate(zip(image_paths, imageset_paths, anno_paths, split_names)):
        image_path = os.path.join(extract_root, image_path)
        imageset_path = os.path.join(extract_root, imageset_path)
        anno_path = os.path.join(extract_root, anno_path)
        imageset_files = [f for f in open(imageset_path)]
        for image_file in imageset_files:
            # copy image file
            image_file = image_file.rstrip()
            src_file = os.path.join(image_path, image_file)+'.jpg'
            dst_file = os.path.join(example_project_files_path, image_file)+'.jpg'
            shutil.copy2(src_file, dst_file)
            # parse annotation file
            anno_file = os.path.join(anno_path, image_file)+'.xml'
            voc_dict = dataset_utils.parse_voc_xml_file(anno_file)

            file_name_partial = dataset_utils.get_file_name_partial(dst_file, state.projects_path)
            image_size = voc_dict['annotation']['size']
            image_info = {
                'id': image_id,
                'license': None,
                'file_name': os.path.basename(file_name_partial),
                'height': int(image_size['height']),
                'width': int(image_size['width']),
                'split_name': split_name
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

