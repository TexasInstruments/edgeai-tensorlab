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
import shutil
from colorama import Fore
from .image_cls import  ImageClassification
from .. import utils


'''
Reference: 
Do ImageNet Classifiers Generalize to ImageNet?
Benjamin Recht∗, Rebecca Roelofs, Ludwig Schmidt, Vaishaal Shankar
https://arxiv.org/abs/1902.10811
Source Code: https://github.com/modestyachts/ImageNetV2'
Download: http://imagenetv2public.s3-website-us-west-2.amazonaws.com/
'''


class ImageNetV2(ImageClassification):
    def __init__(self, *args, num_classes=1000, url=None, download=False, num_frames=None, name=None, **kwargs):
        self.url = url
        self.class_names_dict = None
        self.class_ids_dict = None
        super().__init__(*args, num_classes=num_classes, download=download, num_frames=num_frames, name=name, **kwargs)

    def get_notice(self):
        notice = f'{Fore.YELLOW}' \
                 f'\nThe ImageNetV2 dataset contains new test data for the ImageNet benchmark.' \
                 f'\n             It is smaller in size and faster to download - ' \
                 f'\n             ImageNetV2c closely matches the accuracy obtained with original ImageNet.' \
                 f'\n             So it is a good choice for quick benchmarking.\n' \
                 f'\nReference  : Do ImageNet Classifiers Generalize to ImageNet? ' \
                 f'\n             Benjamin Recht et.al. https://arxiv.org/abs/1902.10811' \
                 f'\nSource Code: https://github.com/modestyachts/ImageNetV2' \
                 f'\nOriginal ImageNet Dataset URL: http://image-net.org' \
                 f'{Fore.RESET}\n'
        return notice

    def download(self, path, split_file):
        print(utils.log_color('\nINFO', 'downloading and preparing dataset', path + ' This may take some time.'))
        print(self.get_notice())
        root = self._get_root(path)
        download_root = os.path.join(root, 'download')
        extract_root = os.path.join(download_root, 'rawdata')
        extract_path = utils.download_file(self.url, root=download_root, extract_root=extract_root, mode='r',
                                           force_download=self.force_download)

        folders = utils.list_dir(os.path.join(extract_path, 'imagenetv2-top-images-format-val'))
        basename_to_int = lambda f:int(os.path.basename(f))
        folders = sorted(folders, key=basename_to_int)
        lines = []
        for folder_id, folder in enumerate(folders):
            src_files = utils.list_files(folder)
            files = [os.path.join(os.path.basename(folder), os.path.basename(f)) for f in src_files]
            dst_files = [os.path.join(path, f) for f in files]
            for src_f, dst_f in zip(src_files, dst_files):
                os.makedirs(os.path.dirname(dst_f), exist_ok=True)
                shutil.copy2(src_f, dst_f)
            #
            folder_lines = [f'{f} {folder_id}' for f in files]
            lines.extend(folder_lines)
        #

        with open(split_file, 'w') as fp:
            fp.write('\n'.join(lines))
        #
        print(utils.log_color('\nINFO', 'dataset ready', path))
        return extract_path, split_file

    def _get_root(self, path):
        path = path.rstrip('/')
        root = os.sep.join(os.path.split(path)[:-1])
        return root


class ImageNetV2A(ImageNetV2):
    def __init__(self, *args, name='imagenetv2c', **kwargs):
        url = 'https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-top-images.tar.gz'
        super().__init__(*args, url=url, name=name, **kwargs)


class ImageNetV2B(ImageNetV2):
    def __init__(self, *args, name='imagenetv2b', **kwargs):
        url = 'https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz'
        super().__init__(*args, url=url, name=name, **kwargs)


class ImageNetV2C(ImageNetV2):
    def __init__(self, *args, name='imagenetv2c', **kwargs):
        url = 'https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-top-images.tar.gz'
        super().__init__(*args, url=url, name=name, **kwargs)
