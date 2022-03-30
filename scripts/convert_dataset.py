#################################################################################
# Copyright (c) 2018-2022, Texas Instruments Incorporated - http://www.ti.com
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
import sys
import json
import glob
import PIL
import PIL.Image
import math
import argparse


def convert_cityscapes(args):
    split_names = ['train', 'val']
    polygon_suffix = '_gtFine_polygons.json'
    source_image_suffix = '.png'
    dest_image_suffix = '.jpg'

    # the background class is anything except the other four classes
    # background class is not fully annotated - so should not be used for training
    # it is basically used to indicated an area that overlaps with a fully annotated class, only whenever that happens.
    classes_selected = ['background', 'road', 'human', 'roadsign', 'vehicle']

    classes_mapping = {
        'background': 'background',
        'unlabeled': 'background',
        'ego vehicle': 'background',
        'rectification border': 'background',
        'out of roi': 'background',
        'static': 'background',
        'dynamic': 'background',
        'ground': 'background',
        'road': 'road',
        'sidewalk': 'background',
        'parking': 'background',
        'rail track': 'background',
        'building': 'background',
        'wall': 'background',
        'fence': 'background',
        'guard rail': 'background',
        'bridge': 'background',
        'tunnel': 'background',
        'pole': 'background',
        'polegroup': 'background',
        'traffic sign': 'roadsign',
        'traffic light': 'roadsign',
        'vegetation': 'background',
        'terrain': 'background',
        'sky': 'background',
        'person': 'human',
        'rider': 'human',
        'car': 'vehicle',
        'truck': 'vehicle',
        'bus': 'vehicle',
        'caravan': 'vehicle',
        'trailer': 'vehicle',
        'train': 'vehicle',
        'motorcycle': 'vehicle',
        'bicycle': 'vehicle',
        'license plate': 'background',
    }
    background_class = 'background'

    ann_data = {}
    annotations = []
    images = []
    image_id = 0
    annotation_id = 0
    category_id_start = 0 #1
    categories = [dict(id=class_id+category_id_start, name=class_name) for class_id, class_name in enumerate(classes_selected)]

    os.makedirs(args.dest_anno, exist_ok=True)

    for split_name in split_names:
        json_files = glob.glob(f'{args.source_anno}/{split_name}/*/*{polygon_suffix}')
        for json_file in json_files:
            with open(json_file) as jfp:
                json_data = json.load(jfp)
            #
            image_file = json_file.replace(polygon_suffix, source_image_suffix).replace(args.source_anno, args.source_data)
            image = PIL.Image.open(image_file)
            image_size = image.size
            image_file = os.path.basename(image_file)
            file_name = f'{os.path.splitext(image_file)[0]}{dest_image_suffix}'
            image_info = {
                'id': image_id,
                'license': None,
                'width': image_size[0],
                'height': image_size[1],
                'file_name': file_name,
                'split_name': split_name
            }
            images.append(image_info)

            for obj in json_data['objects']:
                deleted = obj['deleted']
                if deleted:
                    annotation_id += 1
                    continue
                #
                label = obj['label']
                label_mapped = None
                if label in classes_mapping:
                    label_mapped = classes_mapping[label]
                else:
                    for label_name in classes_mapping:
                        if label_name.startswith(label):
                            label_mapped = classes_mapping[label_name]
                        #
                    #
                #
                label_mapped = label_mapped or label

                polygon = obj['polygon']
                segmentation = []
                bbox_xyxy = [math.inf, math.inf, 0, 0]
                for p in polygon:
                    segmentation.extend([p[0], p[1]])
                    bbox_xyxy[0] = min(bbox_xyxy[0], p[0])
                    bbox_xyxy[1] = min(bbox_xyxy[1], p[1])
                    bbox_xyxy[2] = max(bbox_xyxy[2], p[0])
                    bbox_xyxy[3] = max(bbox_xyxy[3], p[1])
                #
                segmentation = [round(s, 2) for s in segmentation]
                if label_mapped != background_class:
                    bbox_xywh = [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2]-bbox_xyxy[0], bbox_xyxy[3]-bbox_xyxy[1]]
                    bbox_xywh = [round(v, 2) for v in bbox_xywh]
                    area = bbox_xywh[2]*bbox_xywh[3]
                else:
                    bbox_xywh = None
                    area = None
                #
                category_id = classes_selected.index(label_mapped) + category_id_start
                annotation_info = {
                    'bbox': bbox_xywh,
                    'segmentation': segmentation,
                    'area': area,
                    'iscrowd': 0,
                    'image_id': image_id,
                    'category_id': category_id,
                    'id': annotation_id
                }
                annotations.append(annotation_info)
                annotation_id += 1
            #
            image_id += 1
        #
    #
    info = {
        "description": "TIScapes2017 Driving Dataset",
        "url": None,
        "version": "1.0",
        "year": 2017,
        "contributor": "Texas instruments",
        "date_created": "2017/feb/07"
    }
    dataset_store = dict(info=info, categories=categories, images=images, annotations=annotations)
    with open(args.dest_anno, 'w') as jfp:
        json.dump(dataset_store, jfp)
    #


def convert_labelstudio_detection(args):
    dataset_store = dict(info=dict(), categories=list(),
                         images=[], annotations=[])
    with open(args.source_anno) as afp:
        dataset_json_min = json.load(afp)
    #
    # find categories
    all_labels = []
    annotation_id = 0
    for image_info_json_min in dataset_json_min:
        image_labels = [label_info['rectanglelabels'][0] for label_info in image_info_json_min['label']]
        all_labels += image_labels
    #
    categories = list(set(all_labels))
    # add a dummy class so that the index() find category_id will start from 1
    categories_temp = ['background'] + categories
    dataset_store['categories'] = [dict(id=id, name=name) for id, name in enumerate(categories_temp)]
    for image_info_json_min in dataset_json_min:
        # image info
        image_id = image_info_json_min['id']
        basename = os.path.basename(image_info_json_min['image'])
        image_file = os.path.join(args.source_data, basename)
        pil_image = PIL.Image.open(image_file)
        image_size = pil_image.size
        image_info = {"width": image_size[0], "height": image_size[1], "id": image_id, "file_name": basename}
        dataset_store['images'].append(image_info)
        # annotation
        for label_info in image_info_json_min['label']:
            label_name = label_info['rectanglelabels'][0]
            category_id = categories_temp.index(label_name)
            bbox_info = [label_info['x'], label_info['y'], label_info['width'], label_info['height']]
            annotation_info = {"id": annotation_id, "image_id": image_id, "category_id": category_id, "segmentation": [],
                                "bbox": bbox_info, "ignore": 0, "iscrowd": 0, "area": 0}
            dataset_store['annotations'].append(annotation_info)
            annotation_id += 1
        #
    #
    with open(args.dest_anno, 'w') as jfp:
        json.dump(dataset_store, jfp)
    #


def convert_labelstudio_classification(args):
    dataset_store = dict(info=dict(), categories=list(),
                         images=[], annotations=[])
    with open(args.source_anno) as afp:
        dataset_json_min = json.load(afp)
    #
    # find categories
    all_labels = []
    annotation_id = 0
    for image_info_json_min in dataset_json_min:
        image_labels = [image_info_json_min['choice']]
        all_labels += image_labels
    #
    categories = list(set(all_labels))
    # add a dummy class so that the index() find category_id will start from 1
    categories_temp = ['background'] + categories
    dataset_store['categories'] = [dict(id=id, name=name) for id, name in enumerate(categories_temp)]
    for image_info_json_min in dataset_json_min:
        # image info
        image_id = image_info_json_min['id']
        basename = os.path.basename(image_info_json_min['image'])
        image_file = os.path.join(args.source_data, basename)
        pil_image = PIL.Image.open(image_file)

        image_size = pil_image.size
        image_info = {"width": image_size[0], "height": image_size[1], "id": image_id, "file_name": basename}
        dataset_store['images'].append(image_info)
        # annotation
        label_name = image_info_json_min['choice']
        category_id = categories_temp.index(label_name)
        annotation_info = {"id": annotation_id, "image_id": image_id, "category_id": category_id, "segmentation": [],
                           "bbox": None, "ignore": 0, "iscrowd": 0, "area": 0}
        dataset_store['annotations'].append(annotation_info)
        annotation_id += 1
    #
    with open(args.dest_anno, 'w') as jfp:
        json.dump(dataset_store, jfp)
    #


def convert_coco_splits(args):
    dataset_store = dict(info=dict(), categories=list(),
                         images=[], annotations=[])
    source_anno_splits = args.source_anno.split(',')
    source_data_splits = args.source_data.split(',')
    info = None
    categories = None
    images = []
    annotations = []
    max_image_id = 0
    max_annotation_id = 0
    split_names = ('train', 'val')
    for source_data_split, source_anno_split, split_name in zip(source_data_splits, source_anno_splits, split_names):
        with open(source_anno_split) as afp:
            dataset_split = json.load(afp)
            info = dataset_split['info']
            categories = dataset_split['categories']
            for image_info in dataset_split['images']:
                max_image_id = max(image_info['id'], max_annotation_id)
            #
            for annotation_info in dataset_split['annotations']:
                max_annotation_id = max(annotation_info['id'], max_annotation_id)
            #

            for image_info in dataset_split['images']:
                image_info['id'] += max_image_id
                image_info['split_name'] = split_name
            #
            for annotation_info in dataset_split['annotations']:
                annotation_info['id'] += max_annotation_id
            #
            dataset_store['info'] = info
            dataset_store['categories'] = categories
            dataset_store['images'].extend(dataset_split['images'])
            dataset_store['annotations'].extend(dataset_split['annotations'])
        #
    #
    with open(args.dest_anno, 'w') as jfp:
        json.dump(dataset_store, jfp)
    #


def convert_image_folders(args, is_dataset_split=False, supported_image_types=('.jpeg', '.jpg', '.png', '.webp', '.bmp')):
    if is_dataset_split:
        split_folders = ['']
    else:
        split_folders = os.listdir(args.source_data)
    #
    dataset_store = {'info':{}, 'categories':[], 'images':[], 'annotations':[]}
    categories = []
    image_id = 0
    annotation_id = 0
    for split_name in split_folders:
        split_folder_path = os.path.join(args.source_data, split_name)
        categories = sorted(os.listdir(split_folder_path))
        for image_class_id, image_class in enumerate(categories):
            category_id = image_class_id + 1
            image_class_path = os.path.join(split_folder_path, image_class)
            image_files = os.listdir(image_class_path)
            for image_file in image_files:
                image_file_path = os.path.join(image_class_path, image_file)
                if not os.path.splitext(image_file_path)[-1].lower() in supported_image_types:
                    continue
                #
                pil_image = PIL.Image.open(image_file_path)
                image_size = pil_image.size
                image_file_name = f'{image_class}/{image_file}'
                image_info = {"width": image_size[0], "height": image_size[1], "id": image_id, "file_name": image_file_name}
                if split_name != '':
                    image_info["split_name"] = split_name
                #
                dataset_store["images"] += [image_info]
                annotation_info = {"id": annotation_id, "image_id": image_id, "category_id": category_id, "segmentation": [],
                                   "bbox": None, "ignore": 0, "iscrowd": 0, "area": 0}
                dataset_store['annotations'] += [annotation_info]
                annotation_id += 1
                image_id += 1
            #
        #
    #
    dataset_store['info'] = {
        'description': 'Classification Dataset',
        'format': 'COCO 2017, https://cocodataset.org'
    }
    dataset_store['categories'] = [dict(id=cat_id+1, supercategory=cat_name, name=cat_name) for cat_id, cat_name in enumerate(categories)]
    with open(args.dest_anno, 'w') as jfp:
        json.dump(dataset_store, jfp)
    #


def main(args):
    if args.source_format == 'cityscapes':
        convert_cityscapes(args)
    elif args.source_format == 'labelstudio_classification':
        convert_labelstudio_classification(args)
    elif args.source_format == 'labelstudio_detection':
        convert_labelstudio_detection(args)
    elif args.source_format == 'coco_splits':
        convert_coco_splits(args)
    elif args.source_format == 'image_folders':
        convert_image_folders(args)
    elif args.source_format == 'image_splits':
        convert_image_folders(args, is_dataset_split=True)
    else:
        assert False, 'unrecognized source format'


if __name__ == '__main__':
    print(f'argv: {sys.argv}')
    # the cwd must be the root of the repository
    if os.path.split(os.getcwd())[-1] == 'scripts':
        os.chdir('../')
    #

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_data', type=str, default=None)
    parser.add_argument('--source_anno', type=str, default=None)
    parser.add_argument('--source_format', type=str, default=None)
    parser.add_argument('--dest_anno', type=str, default=None)
    args = parser.parse_args()

    main(args)
