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
import urllib
import gzip
import tarfile
import zipfile
import numpy as np
import json
import shutil
import progressbar
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


class ProgressBarUpdater():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()
        #
        cur_size = block_num*block_size
        self.pbar.update(cur_size)
        if cur_size >= total_size:
            self.pbar.finish()
        #


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


def copy_file(file_path, file_path_local):
    os.makedirs(os.path.dirname(file_path_local), exist_ok=True)
    shutil.copy2(file_path, file_path_local)


def extract_files(download_file, extract_root):
    extract_success = False
    if extract_root is not None:
        if download_file.endswith('.tar'):
            with tarfile.open(download_file, 'r') as tar:
                tar.extractall(path=extract_root)
            #
            extract_success = True
        elif download_file.endswith('.tar.gz') or download_file.endswith('.tgz'):
            with tarfile.open(download_file, 'r:gz') as tar:
                tar.extractall(path=extract_root)
            #
            extract_success = True
        elif download_file.endswith('.gz'):
            to_path = os.path.join(extract_root, os.path.splitext(os.path.basename(download_file))[0])
            with open(to_path, "wb") as out_f, gzip.GzipFile(download_file) as zip_f:
                out_f.write(zip_f.read())
            #
            extract_success = True
        elif download_file.endswith('.zip'):
            with zipfile.ZipFile(download_file, 'r') as z:
                z.extractall(extract_root)
            #
            extract_success = True
        #
        return extract_success


def download_and_extract(dataset_url, download_root, extract_root=None, save_filename=None, progressbar_creator=None):
    progressbar_creator = progressbar_creator or ProgressBarUpdater
    download_path = None
    exception_message = ''
    try:
        save_filename = save_filename if save_filename else os.path.basename(dataset_url)
        download_file = os.path.join(download_root, save_filename)
        if not os.path.exists(download_file):
            progressbar_obj = progressbar_creator()
            os.makedirs(download_root, exist_ok=True)
            print(f'downloading from {dataset_url} to {download_file}')
            urllib.request.urlretrieve(dataset_url, download_file, progressbar_obj)
        #
        download_path = download_file
        download_success = True
    except urllib.error.URLError as message:
        download_success = False
        exception_message = str(message)
        print(exception_message)
    except urllib.error.HTTPError as message:
        download_success = False
        exception_message = str(message)
        print(exception_message)
    except Exception as message:
        # sometimes getting exception even though download succeeded.
        download_path = download_file
        download_success = True
        exception_message = str(message)
    #
    if download_success:
        extract_files(download_path, extract_root)
    #
    return download_success, exception_message, download_path


def download_file(dataset_url, download_root, extract_root=None, save_filename=None, progressbar_creator=None, force_linkfile=True):
    if not isinstance(dataset_url, str):
        return False, '', ''

    download_root = os.path.abspath('./') if download_root is None else download_root
    is_url = (dataset_url.startswith('http://') or dataset_url.startswith('https://'))

    if not is_url:
        if dataset_url.endswith('.link'):
            with open(dataset_url) as fp:
                dataset_url = fp.read().rstrip()
            #
        elif force_linkfile and not os.path.exists(dataset_url):
            url_link = dataset_url+'.link'
            if os.path.exists(url_link):
                with open(url_link) as fp:
                    dataset_url = fp.readline().rstrip()
                #
            #
        #
    #
    # update, based on the content of the .link file
    is_url = (dataset_url.startswith('http://') or dataset_url.startswith('https://'))

    if not is_url:
        try:
            if not extract_files(dataset_url, extract_root):
                if os.path.isdir(dataset_url):
                    if os.path.islink(extract_root):
                        os.unlink(extract_root)
                    #
                    os.symlink(dataset_url, extract_root)
                else:
                    return False, '', ''
                #
            #
        except FileNotFoundError:
            return False, '', ''
        #
        return True, '', extract_root
    #

    return download_and_extract(dataset_url, download_root, extract_root=extract_root, save_filename=save_filename,
                                 progressbar_creator=progressbar_creator)


def download_files(dataset_urls, download_root, extract_root=None, save_filenames=None, log_writer=None, progressbar_creator=None):
    if log_writer is not None:
        success_writer, warning_writer = log_writer[:2]
    else:
        success_writer, warning_writer = print, print
    #
    dataset_urls = dataset_urls if isinstance(dataset_urls, (list,tuple)) else [dataset_urls]
    save_filenames = save_filenames if isinstance(save_filenames, (list,tuple)) else \
        ([None]*len(dataset_urls) if save_filenames is None else [save_filenames])

    download_paths = []
    for dataset_url_id, (dataset_url, save_filename) in enumerate(zip(dataset_urls, save_filenames)):
        success_writer(f'Downloading {dataset_url_id+1}/{len(dataset_urls)}: {dataset_url}')
        download_success, message, download_path = download_file(dataset_url, download_root=download_root, extract_root=extract_root,
                                                      save_filename=save_filename, progressbar_creator=progressbar_creator)
        if download_success:
            success_writer(f'Download done for {dataset_url}')
        else:
            warning_writer(f'Download failed for {dataset_url} {str(message)}')
        #
        download_paths.append(download_path)
    #
    return download_success, message, download_paths


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


annotation_prefix_dict = {
    constants.TASK_TYPE_CLASSIFICATION : 'labels',
    constants.TASK_TYPE_DETECTION : 'instances'
}


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
    if isinstance(dataset, str):
        with open(dataset) as fp:
            dataset = json.load(fp)
        #
    #
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
    print('dataset split sizes', image_count_split)
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


def dataset_split_write(input_data_path, dataset_dict, input_data_path_split, annotation_path_split):
    os.makedirs(os.path.dirname(annotation_path_split), exist_ok=True)
    pretty_json_dump(annotation_path_split, dataset_dict)
    return


def dataset_split_link(input_data_path, dataset_dict, input_data_path_split, annotation_path_split):
    if input_data_path is not None:
        if not os.path.exists(input_data_path_split):
            cwd_path = os.getcwd()
            os.chdir(os.path.dirname(input_data_path_split))
            input_data_path_split_base = os.path.basename(input_data_path_split)
            if os.path.islink(input_data_path_split_base):
                os.unlink(input_data_path_split_base)
            #
            if os.path.exists(input_data_path_split_base):
                print(f'{input_data_path_split}: exists')
            else:
                os.symlink(input_data_path, input_data_path_split_base)
                os.chdir(cwd_path)
            #
        #
    #
    return


def dataset_load_coco(task_type, input_data_path, input_annotation_path):
    with open(input_annotation_path) as afp:
        dataset_store = json.load(afp)
    #
    for image_info in dataset_store['images']:
        image_info['file_name'] = os.path.basename(image_info['file_name'])
    #
    return dataset_store


def dataset_load(task_type, input_data_path, input_annotation_path, annotation_format='coco_json', is_dataset_split=False):
    dataset_store = dataset_load_coco(task_type, input_data_path, input_annotation_path)
    return dataset_store
