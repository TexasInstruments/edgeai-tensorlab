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

import copy
import os
import random
import glob
import base64
import collections
import numpy as np
import json


import PIL
from PIL import ImageOps
from xml.etree.ElementTree import Element as ET_Element

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
#
from typing import Any, Callable, Dict, Optional, Tuple, List

from .. import constants
from .. import utils


def parse_voc_xml_file(annotation_file_name: str) -> Dict[str, Any]:
    node = ET_parse(annotation_file_name).getroot()
    return parse_voc_xml(node)


def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
    voc_dict: Dict[str, Any] = {}
    children = list(node)
    if children:
        def_dic: Dict[str, Any] = collections.defaultdict(list)
        for dc in map(parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == "annotation":
            def_dic["object"] = [def_dic["object"]]
        voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict


def get_new_id(id_list):
    found_ids_max = max(id_list) if id_list else 0
    found_gaps = [id for id in range(1, found_ids_max+2) if id not in id_list]
    return found_gaps[0]


def get_category_names(categories_list):
    category_names = [d['name'] for d in categories_list]
    return category_names


def get_category_ids(categories_list):
    category_ids = [d['id'] for d in categories_list]
    return category_ids


def get_category_entry(categories_list, category_name):
    category_entries = [d for d in categories_list if d['name'] == category_name]
    category_entry = category_entries[0] if len(category_entries) > 0 else None
    return category_entry


def get_category_id(categories_list, category_name):
    category_ids = get_category_ids(categories_list)
    category_ids = [c for c in category_ids if c == category_name]
    category_id = category_ids[0] if len(category_ids) > 0 else None
    return category_id


def get_category_name(categories_list, category_id):
    category_names = get_category_names(categories_list)
    category_names = [c for c in category_names if c == category_id]
    category_name = category_names[0] if len(category_names) > 0 else None
    return category_name


def get_new_category_id(categories_list):
    category_ids = get_category_ids(categories_list)
    category_id = get_new_id(category_ids)
    return category_id


def add_missing_categories(categories_list, missing_category_name='undefined'):
    if len(categories_list) == 0:
        return categories_list
    #
    category_ids = [d['id'] for d in categories_list]
    category_ids_max = max(category_ids)
    category_ids_missing = [id for id in range(1,category_ids_max+1) if id not in category_ids]
    categories_list_out = copy.deepcopy(categories_list)
    for category_id in category_ids_missing:
        name = f'{missing_category_name}{category_id}'
        category_entry = dict(id=category_id, supercategory=name, name=name)
        categories_list_out.append(category_entry)
    #
    # sort
    categories_list_out = sorted(categories_list_out, key=lambda d:d['id'])
    return categories_list_out


def adjust_categories(categories_list, category_names_new, missing_category_name='undefined'):
    categories_list_out = []
    for category_name in category_names_new:
        category_entry = get_category_entry(categories_list, category_name)
        if category_entry is None:
            new_category_id = get_new_category_id(categories_list)
            category_entry = dict(id=new_category_id, supercategory=category_name, name=category_name)
        #
        categories_list_out.append(category_entry)
    #
    categories_list_out = add_missing_categories(categories_list_out, missing_category_name)
    return categories_list_out


def get_file_list(dataset_path):
    file_list = glob.glob(f'{dataset_path}/*.*')
    return file_list


def get_file_name_from_partial(dataset_file_partial, project_path):
    file_name = os.path.join(project_path, dataset_file_partial)
    return file_name


def get_file_name_partial(dataset_file, project_path):
    file_name_partial = dataset_file.replace(project_path + os.sep, '') if dataset_file else None
    return file_name_partial


def get_file_names_partial(dataset_files, project_path):
    file_names_partial = [get_file_name_partial(f, project_path) for f in dataset_files]
    return file_names_partial


# Author: Manu Mathew
# Date: 2021 March
def get_color_table(num_classes):
    num_classes_3 = np.power(num_classes, 1.0/3)
    delta_color = int(256/num_classes_3)
    colors = [(r, g, b) for r in range(0,256,delta_color)
                        for g in range(0,256,delta_color)
                        for b in range(0,256,delta_color)]
    # spread the colors list to num_classes
    color_step = len(colors) / num_classes
    colors_list = []
    to_idx = 0
    while len(colors_list) < num_classes:
        from_idx = round(color_step * to_idx)
        if from_idx < len(colors):
            colors_list.append(colors[from_idx])
        else:
            break
        #
        to_idx = to_idx + 1
    #
    shortage = num_classes-len(colors_list)
    if shortage > 0:
        colors_list += colors[-shortage:]
    #
    r_list = [c[0] for c in colors_list]
    g_list = [c[1] for c in colors_list]
    b_list = [c[2] for c in colors_list]
    max_color = (max(r_list), max(g_list), max(b_list))
    color_offset = ((255-max_color[0])//2, (255-max_color[1])//2, (255-max_color[2])//2)
    colors_list = [(c[0]+color_offset[0], c[1]+color_offset[1], c[2]+color_offset[2]) for c in colors_list]
    return colors_list


def get_color_palette(num_classes):
    colors_list = get_color_table(num_classes)
    if len(colors_list) < 256:
        colors_list += [(255,255,255)] * (256-len(colors_list))
    #
    assert len(colors_list) == 256, f'incorrect length for color palette {len(colors_list)}'
    return colors_list


def get_file_as_url(file_name):
    # streamlit can serve file content directly in base64 format
    # Note: in cases where it cannot, we will need to run an external fileserver
    file_url = None
    if file_name is not None:
        with open(file_name, 'rb') as fp:
            file_buffer = fp.read()
            file_content = base64.b64encode(file_buffer).decode('utf-8')
            file_ext = os.path.splitext(file_name)[-1]
            file_ext = file_ext[1:] if len(file_ext) > 0 else file_ext
            file_url = f'data:image/{file_ext};base64,{file_content}'
        #
    #
    return file_url


def get_file_as_image(file_name):
    return PIL.Image.open(file_name) if file_name else None


def resize_image(image, output_width=None, output_height=None, with_pad=False):
    if isinstance(image, str):
        image = PIL.Image.open(image)
    #
    border = (0,0,0,0)
    resize_width = output_width
    resize_height = output_height
    if resize_width is None and resize_height is None:
        return image, border
    #
    input_width, input_height = image.size
    input_ratio = input_width / input_height
    output_ratio = output_width / output_height
    if resize_width is None or (with_pad and output_ratio >= input_ratio):
        # pad width
        resize_width = round(input_width * resize_height / input_height)
    elif resize_height is None or (with_pad and output_ratio < input_ratio):
        # pad height
        resize_height = round(input_height * resize_width / input_width)
    #
    image = image.resize((resize_width, resize_height))
    wpad = round(output_width - resize_width)
    hpad = round(output_height - resize_height)
    top = hpad // 2
    bottom = hpad - top
    left = wpad // 2
    right = wpad - left
    border = (left, top, right, bottom)
    image = ImageOps.expand(image, border=border, fill=0)
    return image, border


def pretty_json_dump(file_name, data):
    has_float_repr = False
    if hasattr(json.encoder, 'FLOAT_REPR'):
        has_float_repr = True
        float_repr_backup = json.encoder.FLOAT_REPR
    #
    json.encoder.FLOAT_REPR = lambda x: f'{x:g}'
    with open(file_name, 'w') as fp:
        json.dump(data, fp)
    #
    if has_float_repr:
        json.encoder.FLOAT_REPR = float_repr_backup
    else:
        encoder = json.encoder
        del encoder.FLOAT_REPR


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


def dataset_split(dataset, split_factor, split_names, random_seed=1):
    random.seed(random_seed)
    dataset_train = dict(info=dataset['info'],
                         categories=dataset['categories'],
                         images=[], annotations=[])
    dataset_val = dict(info=dataset['info'],
                         categories=dataset['categories'],
                         images=[], annotations=[])
    dataset_splits = {split_names[0]:dataset_train, split_names[1]:dataset_val}

    annotations_info_list = _find_annotations_info(dataset)
    image_count_split = {split_name:0 for split_name in split_names}
    for image_id, (image_info, annotations) in enumerate(zip(dataset['images'], annotations_info_list)):
        if not annotations:
            # ignore images without annotations from the splits
            continue
        #
        image_info['file_name'] = os.path.basename(image_info['file_name'])
        if 'split_name' in image_info and image_info['split_name'] is not None:
            # print(f'file_name={image_info["file_name"]} split_name={image_info["split_name"]}')
            split_name = image_info['split_name']
            split_name = split_names[0] if 'train' in split_name else split_name #change trainval to tarin
            split_name = split_names[1] if 'test' in split_name else split_name  #change test to val
        else:
            # print(f'split_name was not found in {image_info["file_name"]}')
            split_name = split_names[0] if random.random() < split_factor else split_names[1]
        #
        dataset_splits[split_name]['images'].append(image_info)
        dataset_splits[split_name]['annotations'].extend(annotations)
        image_count_split[split_name] += 1
    #
    print('INFO: ModelMaker - dataset split sizes', image_count_split)
    return dataset_splits


def dataset_split_limit(dataset_dict, max_num_files):
    if max_num_files is None:
        return dataset_dict
    #
    annotations_info_list = _find_annotations_info(dataset_dict)
    dataset_new = dict(info=dataset_dict['info'], categories=dataset_dict['categories'],
                       images=[], annotations=[])
    for image_id, (image_info, annotations) in enumerate(zip(dataset_dict['images'], annotations_info_list)):
        if image_id >= max_num_files:
            break
        #
        dataset_new['images'].append(image_info)
        dataset_new['annotations'].extend(annotations)
    #
    return dataset_new


def dataset_split_write(input_data_path, dataset_dict, input_data_path_split,
                        annotation_path_split):
    os.makedirs(os.path.dirname(annotation_path_split), exist_ok=True)
    pretty_json_dump(annotation_path_split, dataset_dict)
    return


def dataset_split_link(input_data_path, dataset_dict, input_data_path_split, annotation_path_split):
    utils.make_symlink(input_data_path, input_data_path_split)
    return


def dataset_load_coco(input_annotation_path, input_data_path=None, task_type=None):
    with open(input_annotation_path) as afp:
        dataset_store = json.load(afp)
    #
    for image_info in dataset_store['images']:
        image_info['file_name'] = os.path.basename(image_info['file_name'])
    #
    return dataset_store


def dataset_load(input_annotation_path, input_data_path=None, task_type=None, annotation_format='coco_json',
                 is_dataset_split=False, fix_errors=False):
    dataset_store = dataset_load_coco(input_annotation_path, input_data_path, task_type)
    image_ids = [image_info['id'] for image_info in dataset_store['images']]
    if (any([isinstance(image_id, str) for image_id in image_ids]) and
            any([isinstance(image_id, int) for image_id in image_ids])):
        if fix_errors:
            print('WARNING: incorrect dataset format - found a mix of int and string image_id - '
                  'this can cause a crash during dataset loading - changing to ints.')
            original_to_new_image_id_dict = dict()
            for image_id, image_info in enumerate(dataset_store['images']):
                new_image_id = image_id + 1
                orig_image_id = image_info['id']
                image_info['id'] = new_image_id
                image_info['id_orig'] = orig_image_id
                original_to_new_image_id_dict[orig_image_id] = new_image_id
            #
            for anno_id, anno in enumerate(dataset_store['annotations']):
                anno['image_id'] = original_to_new_image_id_dict[anno['image_id']]
            #
        else:
            print('ERROR: incorrect dataset format - found a mix of int and string image_id - '
                  'this can cause a crash during dataset loading.')
            return None
        #
    #
    return dataset_store
