# Copyright (c) 2018-2021, Texas Instruments
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

#################################################################################


"""
Reference:

WIDER FACE: A Face Detection Benchmark,
Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou,
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
http://shuoyang1213.me/WIDERFACE/
"""


import numbers
import os
import random
import json
import shutil
import tempfile
import numpy as np
from colorama import Fore
import datetime
import PIL

from .. import utils
from .dataset_base import *
from . import coco_det

__all__ = ['WiderFaceDetection', 'widerfacedet_det_label_offset_1to1']


class WiderFaceDetection(coco_det.COCODetection):
    def __init__(self, num_classes=1, download=False, image_dir=None, annotation_file=None, verbose_mode=True,
                 num_frames=None, name='widerface', **kwargs):
        self.verbose_mode = verbose_mode
        if image_dir is None or annotation_file is None:
            self.force_download = True if download == 'always' else False
            assert 'path' in kwargs and 'split' in kwargs, 'kwargs must have path and split'
            path = kwargs['path']
            split = kwargs['split']
            if download:
                self.download(path, split)
            #

            dataset_folders = os.listdir(kwargs['path'])
            assert 'annotations' in dataset_folders, 'invalid path to coco dataset annotations'
            annotations_dir = os.path.join(kwargs['path'], 'annotations')

            image_base_dir = 'images' if ('images' in dataset_folders) else ''
            image_base_dir = os.path.join(kwargs['path'], image_base_dir)
            image_split_dirs = os.listdir(image_base_dir)
            assert kwargs['split'] in image_split_dirs, f'invalid path to coco dataset images/split {kwargs["split"]}'
            self.image_dir = os.path.join(image_base_dir, kwargs['split'])
            self.annotation_file = os.path.join(annotations_dir, f'instances_{kwargs["split"]}.json')
        else:
            self.image_dir = image_dir
            self.annotation_file = annotation_file
        #
        with open(self.annotation_file) as afp:
            dataset_store = json.load(afp)
        #
        categories = dataset_store['categories']
        num_classes = num_classes or len(num_classes)
        super().__init__(num_classes=num_classes, num_frames=num_frames, name=name, **kwargs)
        self._load_dataset()
        # create the dataset info
        categories = [{'id':1, 'name':'face'}]
        info = dict(description='WIDERFACE: A Face Detection Dataset', url='http://shuoyang1213.me/WIDERFACE/', version='1.5',
                    year='2017', contributor='Multimedia Laboratory, Department of Information Engineering, The Chinese University of Hong Kong',
                    date_created='2017/mar/31')
        self.dataset_store = dict(info=info, categories=categories)
        self.kwargs['dataset_info'] = self.get_dataset_info()

    def get_categories(self, project_path):
        widerface_categories = [dict(id=1, supercategory='face', name='face')]
        return widerface_categories

    def download(self, path, split, force_download=False):
        root = path
        images_folder = os.path.join(path, split)
        annotations_folder = os.path.join(path, 'annotations')
        if (not self.force_download) and os.path.exists(path) and \
                os.path.exists(images_folder) and os.path.exists(annotations_folder):
            print(utils.log_color('\nINFO', 'dataset exists - will reuse', path))
            return
        #
        print(utils.log_color('\nINFO', 'downloading and preparing dataset', path + ' This may take some time.'))
        print(f'{Fore.YELLOW}'
              f'WIDER FACE: A Face Detection Benchmark\n',
              f'    Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou\n',
              f'    IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016\n',
              f'    http://shuoyang1213.me/WIDERFACE/\n',
              f'{Fore.RESET}\n')

        max_annotations_per_image = 1000
        image_files_path = os.path.join(root, split)
        annotations_path = os.path.join(root, 'annotations')
        annotation_file_name = os.path.join(annotations_path, f'instances_{split}.json')
        extract_root = os.path.join(root, 'other', 'extract')
        download_root = os.path.join(root, 'other', 'download')

        if (not force_download) and os.path.exists(extract_root) and \
            os.path.exists(image_files_path) and os.path.exists(annotations_path) and \
            os.path.exists(annotation_file_name):
            print(f'Dataset exists - will reuse: {root}')
            return (image_files_path,annotation_file_name)
        #

        # * Images URL: https://drive.google.com/file/d/15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M/view?usp=sharing, https://data.deepai.org/widerface.zip
        # * Annotations URL: http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip
        dataset_urls = ['https://data.deepai.org/widerface.zip',
                        'http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip']

        utils.download_files(dataset_urls, root=download_root, extract_root=extract_root)

        # the extracted files still contains some zip - extract them as well
        utils.download_file(os.path.join(extract_root, 'WIDER_train.zip'), root=extract_root, extract_root=extract_root)
        utils.download_file(os.path.join(extract_root, 'WIDER_val.zip'), root=extract_root, extract_root=extract_root)

        categories_list = self.get_categories(root)
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")

        image_paths = ['WIDER_train/images', 'WIDER_val/images']
        anno_paths = ['wider_face_split/wider_face_train_bbx_gt.txt', 'wider_face_split/wider_face_val_bbx_gt.txt']
        split_names = ['train', 'val']
        image_annotations = {split_name:{} for split_name in split_names}
        image_file = None
        num_annotations = 0
        image_id = 0
        for image_path, anno_path, split_name in zip(image_paths, anno_paths, split_names):
            images_folder = os.path.join(path, split_name)
            annotations_folder = os.path.join(path, 'annotations')
            # prepare the dataset folder
            os.makedirs(images_folder, exist_ok=True)
            os.makedirs(annotations_folder, exist_ok=True)
            # create the dataset store
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

            images = []
            annotations = []
            image_annotations_split = image_annotations[split_name]
            image_count_split = 0
            for image_file, anno_bboxes in image_annotations_split.items():
                image_file_basename = os.path.basename(image_file)
                image_file_path = os.path.join(extract_root, image_path, image_file)
                shutil.copy2(image_file_path, os.path.join(images_folder, image_file_basename))
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
                    category_id = categories_list[0]['id'] # there is only one category
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
            if self.verbose_mode:
                print(f'dataset load: split_name={split_name} image_count={image_count_split}')
            #
            dataset_store['annotations'] = annotations
            dataset_store['images'] = images
            annotations_file = os.path.join(annotations_folder, f'instances_{split_name}.json')
            with open(annotations_file, 'w') as afp:
                json.dump(dataset_store, afp)
            #
        #
        print(utils.log_color('\nINFO', 'dataset ready', path))
        return

    def _get_root(self, path):
        path = path.rstrip('/')
        root = os.sep.join(os.path.split(path)[:-1])
        return root

    def get_dataset_info(self):
        if 'dataset_info' in self.kwargs:
            return self.kwargs['dataset_info']
        #
        # return only info and categories for now as the whole thing could be quite large.
        dataset_store = dict()
        for key in ('info', 'categories'):
            if key in self.dataset_store.keys():
                dataset_store.update({key: self.dataset_store[key]})
            #
        #
        dataset_store.update(dict(color_map=self.get_color_map()))        
        return dataset_store


def widerfacedet_det_label_offset_1to1(label_offset=1, num_classes=1):
    coco_label_table = range(1,num_classes+1)
    if label_offset == 1:
        # 0 => 1, 1 => 2, .. 90 => 91
        coco_label_offset = {k:v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({-1:0,num_classes:(num_classes+1)})
    elif label_offset == 0:
        # 0 => 0, 1 => 1, .. 90 => 90
        coco_label_offset = {(k+1):v for k,v in enumerate(coco_label_table)}
        coco_label_offset.update({-1:-1,0:0})
    else:
        assert False, f'unsupported value for label_offset {label_offset}'
    #
    return coco_label_offset

