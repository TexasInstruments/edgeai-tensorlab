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
import PIL

from .... import utils
from . import dataset_utils


widerface_categories = [dict(id=1, supercategory='face', name='face')]


def get_categories(state, example_project_path):
    return widerface_categories


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
        WIDER FACE: A Face Detection Benchmark 
        * Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou
        * IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
        * Dataset URL: http://shuoyang1213.me/WIDERFACE/
        ''')
    # * Images URL: https://drive.google.com/file/d/15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M/view?usp=sharing, https://data.deepai.org/widerface.zip
    # * Annotations URL: http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip

    extract_root = os.path.join(example_project_path, 'other', 'extract')
    if (not force_download) and os.path.exists(extract_root) and \
        os.path.exists(example_project_files_path) and os.path.exists(example_project_annotations_path) and \
        os.path.exists(example_project_annotation_file_name):
        print(f'Dataset exists - will reuse: {example_project_path}')
        return (example_project_files_path,example_project_annotation_file_name)
    #

    dataset_urls = ['https://data.deepai.org/widerface.zip',
                    'http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip']

    download_root = os.path.join(example_project_path, 'other', 'download')
    utils.download_files(dataset_urls, download_root=download_root, extract_root=extract_root,
                                 log_writer=log_writer, progressbar_creator=progressbar_creator)

    # the extracted files still contains some zip - extract them as well
    utils.download_file(os.path.join(extract_root, 'WIDER_train.zip'), download_root=extract_root, extract_root=extract_root)
    utils.download_file(os.path.join(extract_root, 'WIDER_val.zip'), download_root=extract_root, extract_root=extract_root)

    # prepare the dataset folder
    os.makedirs(example_project_files_path, exist_ok=True)
    os.makedirs(example_project_annotations_path, exist_ok=True)

    categories_list = get_categories(state, example_project_path)
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")

    dataset_store = dict()
    dataset_store['info'] = {
        'description': 'WIDER FACE Dataset',
        'version': '0.1',
        'year': '2007 2012',
        'contributor': 'Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou',
        'data_created': date,
        'url': 'http://shuoyang1213.me/WIDERFACE/',
        'format': 'COCO 2017, https://cocodataset.org'
    }
    dataset_store['categories'] = categories_list
    dataset_store['licenses'] = None
    dataset_store['images'] = []
    dataset_store['annotations'] = []

    image_paths = ['WIDER_train/images', 'WIDER_val/images']
    anno_paths = ['wider_face_split/wider_face_train_bbx_gt.txt', 'wider_face_split/wider_face_val_bbx_gt.txt']
    split_names = ['train', 'val']
    image_annotations = {split_name:{} for split_name in split_names}
    image_file = None
    num_annotations = 0
    for image_path, anno_path, split_name in zip(image_paths, anno_paths, split_names):
        image_annotations_split = image_annotations[split_name]
        anno_path = os.path.join(extract_root, anno_path)
        anno_fileid = open(anno_path)
        for line in anno_fileid:
            line = line.strip()
            words = line.split()
            if line.endswith('.jpg'):
                image_file = line
                image_annotations_split[image_file] = []
                line = next(anno_fileid)
                line = line.strip()
                words = line.split()
                num_annotations = int(words[0])
            else:
                if line == '0 0 0 0 0 0 0 0 0 0':
                    continue
                #
                assert num_annotations > 0, f'incorrect format {line} - expected num_annotations > 0'
                assert len(words) > 1, f'incorrect format {line} - expected bbox coordinates'
                anno_bbox = [float(w) for w in words][:4]
                image_annotations_split[image_file].append(anno_bbox)
            #
        #
    #

    image_id = 0
    images = []
    annotations = []
    for image_path, anno_path, split_name in zip(image_paths, anno_paths, split_names):
        image_annotations_split = image_annotations[split_name]
        image_count_split = 0
        for image_file, anno_bboxes in image_annotations_split.items():
            image_file_basename = os.path.basename(image_file)
            image_file_path = os.path.join(extract_root, image_path, image_file)
            shutil.copy2(image_file_path, os.path.join(example_project_files_path, image_file_basename))
            image_size = PIL.Image.open(image_file_path).size
            image_info = {
                'id': image_id,
                'license': None,
                'file_name': image_file_basename,
                'width': image_size[0],
                'height': image_size[1],
                'split_name': split_name
            }
            images.append(image_info)

            annotations_img = []
            anno_bboxes = anno_bboxes[:max_annotations_per_image]
            for object_anno_id, object_anno in enumerate(anno_bboxes):
                category_id = widerface_categories[0]['id'] # there is only one category
                # object_anno: x, y, width, height
                annotation_info = {
                    'id': image_id * max_annotations_per_image + object_anno_id,
                    'segmentation': [],
                    'area': object_anno[2]*object_anno[3],
                    'iscrowd': 0,
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': object_anno[:4]
                }
                annotations_img.append(annotation_info)
            #
            annotations.extend(annotations_img)
            image_count_split += 1
            image_id += 1
        #
        if state.common.verbose_mode:
            print(f'dataset load: split_name={split_name} image_count={image_count_split}')
        #
    #
    dataset_store['annotations'] = annotations
    dataset_store['images'] = images

    with open(example_project_annotation_file_name, 'w') as fp:
        json.dump(dataset_store, fp)
    #
    return (example_project_files_path,example_project_annotation_file_name)
