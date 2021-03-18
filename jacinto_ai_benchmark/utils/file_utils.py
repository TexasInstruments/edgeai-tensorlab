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

import os
import PIL
import numpy as np
from . import download_utils


def get_data_list(input, dest_dir=None, image_formats=None):
    image_formats = ('.png', '.jpg', '.jpeg', '.gif', '.bmp') \
        if image_formats is None else image_formats

    # inData can also be a list/tuple/dict
    input = abspath(input)

    in_files = None
    if isinstance(input, str):
        file_ext = os.path.splitext(input)[1]
        if file_ext.lower() in image_formats:
            in_files = [input]
        elif os.path.isdir(input):
            in_files = os.listdir(input)
            in_files = [os.path.join(input, inp) for inp in in_files]
        elif file_ext.lower() == '.txt':
            with open(input) as input_fp:
                in_files = [line.rstrip() for line in input_fp]
                in_files = [line for line in in_files if (line != '' and line != ' ')]
            #
        else:
            assert False, 'unrecognized input format'
        #
    elif isinstance(input, (list,tuple)):
        in_files = input
    elif isinstance(input, dict):
        path = input.get('path','')
        list_file = input['split']
        with open(list_file) as list_fp:
            in_files = [row.rstrip() for row in list_fp]
        #
        in_files = [os.path.join(path, f) for f in in_files]
    elif isinstance(input, np.ndarray):
        assert dest_dir is not None, 'dest_dir should be provided to save ndarray and return it in a list'
        input_dir = os.path.join(dest_dir, 'in_data')
        os.makedirs(input_dir, exist_ok=True)
        image_name = os.path.join(input_dir, '0.png')
        input = input.astype(np.uint8)
        input = PIL.Image.fromarray(input)
        input.save(image_name)
        in_files = [image_name]
    else:
        assert False, 'unrecognized input format'
    #
    return in_files


def import_data(input, base_dir=None, in_data_list_file='in_data.txt', image_formats=None):
    in_files = get_data_list(input, base_dir, image_formats)
    num_frames = len(in_files)
    if base_dir is not None:
        in_data_list_file = os.path.join(base_dir, in_data_list_file)
    #
    write_list_to_file(in_files, in_data_list_file, base_dir)
    return in_data_list_file, num_frames


def is_weblink(f):
    return isinstance(f,str) and (f.startswith('http://') or f.startswith('https://'))


def abspath(input):
    # inData can also be a list/tuple with first entry being the image path and second and image lsit file
    if isinstance(input, (list, tuple)):
        input = [abspath(input_pth) if not is_weblink(input_pth) else input_pth for input_pth in input]
    elif isinstance(input, dict):
        for p in input:
            input_pth = input[p]
            input[p] = os.path.abspath(input_pth) if (isinstance(input_pth, str) and input_pth.startswith('./')) else input_pth
        #
    elif isinstance(input, str):
        if is_weblink(input):
            pass
        else:
            input = os.path.abspath(input)
        #
    #
    return input


def dict_to_file(kwargs, file_name):
    assert isinstance(kwargs, dict), 'the input must be a dict'
    with open(file_name, 'w') as fp:
        for arg_name in kwargs:
            if kwargs[arg_name] is not None:
                arg = kwargs[arg_name]
                if isinstance(arg, str) and (arg.startswith('./') or arg.startswith('../')):
                    arg = os.path.abspath(arg)
                elif isinstance(arg, (list,tuple)):
                    arg = ' '.join([str(a) for a in arg])
                elif isinstance(arg, dict):
                    arg_file_name = os.path.splitext(file_name)[0] + '_' + arg_name + '.cfg'
                    dict_to_file(arg, arg_file_name)
                    arg = arg_file_name
                #
                fp.write(f'{arg_name} = {arg}\n')
            #
        #
    #


def recursive_listdir(path, ext_list=None):
    filenames = []
    for root, d_names, f_names in os.walk(path):
        for f in f_names:
            filenames.append(os.path.join(root, f))
        #
    #
    if ext_list is not None:
        filenames = [f for f in filenames if os.path.splitext(f)[1] in ext_list]
    #
    return filenames


def write_list_to_file(input_list, file_name, root=None):
    if root is not None:
        file_name = os.path.join(root, file_name)
    #
    with open(file_name, 'w') as list_file_fp:
        for inp in input_list:
            if is_weblink(inp):
                assert root is not None, 'root must be provided to accept http links'
                inp = download_utils.download_url(inp, os.path.join(root, 'in_data'))
            #
            if isinstance(inp, (list,tuple)):
                list_file_fp.write(' '.join(inp) + '\n')
            else:
                list_file_fp.write(f'{inp}\n')
            #
        #
    #


def count_lines(file_name):
    try:
        with open(file_name) as fp:
            return sum(1 for line in fp)
    except:
        print(f"list file {file_name} could not be opened")
        return 0


def get_indata_list(indata_file):
    with open(indata_file, "r") as indata_fp:
        frame_list = list(indata_fp)
        frame_list = [f.rstrip() for f in frame_list]
        return frame_list


def list_dir(d, only_files=False, basename=False):
    files = os.listdir(d)
    files = [os.path.join(d, f) for f in files]
    if only_files:
        files = [f for f in files if os.path.isfile(f)]
    #
    if basename:
        files = [os.path.basename(f) for f in files]
    #
    return files


def list_files(d, basename=False):
    return list_dir(d, only_files=True, basename=basename)

