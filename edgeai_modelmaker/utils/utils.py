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


def import_file_or_folder(folder_or_file_name):
    if folder_or_file_name.endswith(os.sep):
        folder_or_file_name = folder_or_file_name[:-1]
    #
    if folder_or_file_name.endswith('.py'):
        folder_or_file_name = folder_or_file_name[:-3]
    #
    parent_folder = os.path.dirname(folder_or_file_name)
    basename = os.path.basename(folder_or_file_name)
    sys.path.insert(0, parent_folder)
    imported_module = importlib.import_module(basename, __name__)
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
