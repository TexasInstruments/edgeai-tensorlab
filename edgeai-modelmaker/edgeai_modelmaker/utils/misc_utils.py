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
import importlib
import json
import yaml
import tqdm
import errno
import re
import shutil

from . import config_dict


def _absolute_path(relpath):
    if relpath is None:
        return relpath
    elif relpath.startswith('http://') or relpath.startswith('https://'):
        return relpath
    else:
        return os.path.abspath(os.path.expanduser(os.path.normpath(relpath)))


def absolute_path(relpath):
    if isinstance(relpath, (list,tuple)):
        return [_absolute_path(f) for f in relpath]
    else:
        return _absolute_path(relpath)


def remove_if_exists(path):
    try:
        os.remove(path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


def make_symlink(source, dest):
    if source is None or (not os.path.exists(source)):
        print(f'make_symlink failed - source: {source} is invalid')
        return
    #
    remove_if_exists(dest)
    if os.path.dirname(source) == os.path.dirname(dest):
        base_dir = os.path.dirname(source)
        cur_dir = os.getcwd()
        os.chdir(base_dir)
        os.symlink(os.path.basename(source), os.path.basename(dest))
        os.chdir(cur_dir)
    else:
        os.symlink(source, dest)
    #


def import_file_or_folder(folder_or_file_name, package_name=None, force_import=False):
    if folder_or_file_name.endswith(os.sep):
        folder_or_file_name = folder_or_file_name[:-1]
    #
    if folder_or_file_name.endswith('.py'):
        folder_or_file_name = folder_or_file_name[:-3]
    #
    parent_folder = os.path.dirname(folder_or_file_name)
    basename = os.path.basename(folder_or_file_name)
    if force_import:
        sys.modules.pop(basename, None)
    #
    sys.path.insert(0, parent_folder)
    imported_module = importlib.import_module(basename, package_name or __name__)
    sys.path.pop(0)
    return imported_module


def simplify_dict(in_dict):
    '''
    simplify dict so that it can be written using yaml(pyyaml) package
    '''
    assert isinstance(in_dict, (dict, config_dict.ConfigDict)), 'input must of type dict or ConfigDict'
    d = dict()
    for k, v in in_dict.items():
        if isinstance(v, (dict,config_dict.ConfigDict)):
            d[k] = simplify_dict(v)
        elif isinstance(v, tuple):
            d[k] = list(v)
        else:
            d[k] = v
        #
    #
    return d


def write_dict(dict_obj, filename, write_json=True, write_yaml=True):
    if write_json:
        filename_json = os.path.splitext(filename)[0] + '.json'
        with open(filename_json, 'w') as fp:
            json.dump(dict_obj, fp, indent=2, separators=[',',':'])
        #
    #
    if write_yaml:
        dict_obj = simplify_dict(dict_obj)
        filename_yaml = os.path.splitext(filename)[0] + '.yaml'
        with open(filename_yaml, 'w') as fp:
            yaml.safe_dump(dict_obj, fp)
        #
    #


def cleanup_special_chars(file_name):
    if os.path.exists(file_name):
        with open(file_name) as rfp:
            new_lines = []
            log_lines = rfp.readlines()
            for log_line in log_lines:
                log_line = re.sub(r'(\x9B|\x1B[\[\(\=])[0-?]*[ -\/]*([@-~]|$)', '', log_line)
                new_lines.append(log_line)
            #
            with open(file_name, 'w') as wfp:
                wfp.writelines(new_lines)
            #
        #
    #


def sort_annotations(json_file_path, preference_order=None, annotation_prefix="result"):
    """
        Sorting the segmentation annotations to create correct mask.
    """
    if preference_order is None or json_file_path is None:
        sys.exit(
            "Please provide the preference order(increasing order of importance) / Annotation file path for this functionality")

    if preference_order == "" or json_file_path == "":
        sys.exit("Preference order / Annotation file path can't be an empty string")

    preference_order = preference_order.split(",")

    # Loading the json file data if it exists
    if os.path.exists(json_file_path):
        with open(json_file_path) as afp:
            dataset_store = json.load(afp)
        #
    #

    # Checking for the min index
    minval = 1e18
    for i in range(len(dataset_store['categories'])):
        minval = min(minval, dataset_store['categories'][i]['id'])
    #

    # Modify the ID values to 1-indexing if it is zero indexing
    if minval == 0:
        cat_starting_id = 1
        for i in range(len(dataset_store['categories'])):
            dataset_store['categories'][i]['id'] += cat_starting_id
        #

        for i in range(len(dataset_store['annotations'])):
            dataset_store['annotations'][i]['category_id'] += cat_starting_id
        #
    #

    # re-assigning the JSON categories id values based on preference order
    cat_id_mappings = dict()
    for i in range(len(dataset_store['categories'])):
        old_id = dataset_store['categories'][i]['id']
        new_id = preference_order.index(dataset_store['categories'][i]['name']) + 1
        cat_id_mappings[old_id] = new_id
        dataset_store['categories'][i]['id'] = new_id
    #

    # sorting in descending order of JSON categories based on new ids
    dataset_store['categories'].sort(key=lambda x: x['id'])

    # re-assigning JSON annotations category_id values based on preference order
    for i in range(len(dataset_store['annotations'])):
        old_id = dataset_store['annotations'][i]['category_id']
        new_id = cat_id_mappings[old_id]
        dataset_store['annotations'][i]['category_id'] = new_id
    #

    # sorting in descending order of JSON annotations based on new category_ids
    dataset_store['annotations'].sort(key=lambda x: x['category_id'])

    # updating the annotations JSON file
    with open(json_file_path, "w") as afp:
        json.dump(dataset_store, afp)
    #


def reformat_to_modelmaker(input_dataset_path=None):
    """
        Reformatting the folders and files structure similar to modelmaker format
        which is obtained from Label Studio
    """
    if input_dataset_path is None or input_dataset_path == "":
        sys.exit("Please provide the dataset path for reformatting the folder structure")

    # Fetching the destination paths
    destination_folder = os.path.join(input_dataset_path, "annotations")
    destination_path = os.path.join(destination_folder, 'instances.json')

    # Making destination folder if it doesn't exits
    if not os.path.exists(destination_folder):
        # Creating the annotations directory
        os.makedirs(destination_folder)

        # Copying the label studio annotation file to destination place and deleting the older file
        input_annotation_path = os.path.join(input_dataset_path, 'result.json')
        shutil.copyfile(input_annotation_path, destination_path)
        os.remove(input_annotation_path)

    # Checking the existence of the annotation file
    if os.path.exists(destination_path):
        with open(destination_path) as afp:
            dataset_store = json.load(afp)
        #
    #

    # Verifying the indexing whether it is 0 or not
    minval = 1e18
    for i in range(len(dataset_store['categories'])):
        minval = min(minval, dataset_store['categories'][i]['id'])
    #

    if minval == 0:
        # Making the 0-indexing to 1-indexing for category ids in categories
        category_start_id = 1
        for i in range(len(dataset_store['categories'])):
            dataset_store['categories'][i]['id'] += category_start_id
        #

        # Making the 0-indexing to 1-indexing for category ids in annotations
        for i in range(len(dataset_store['annotations'])):
            dataset_store['annotations'][i]['category_id'] += category_start_id
        #

        # updating the annotations JSON file
        with open(destination_path, "w") as afp:
            json.dump(dataset_store, afp)
        #


def is_url(download_entry):
    return isinstance(download_entry, str) and \
            (download_entry.startswith('http://') or download_entry.startswith('https://'))


class ProgressBar():
    def __init__(self, total_size, unit=None):
        self.total_size = total_size
        self.pbar = None
        self.unit = unit

    def __call__(self, cur_size):
        if self.pbar is None:
            # creation of pbar is delayed so that if the call happens in a different process, it will still work
            self.pbar = tqdm.tqdm(total=self.total_size, unit=self.unit)
        #
        self.pbar.update(cur_size)
        if cur_size >= self.total_size:
            self.pbar.close()
        #

    def update(self, cur_size):
        self.__call__(cur_size)
