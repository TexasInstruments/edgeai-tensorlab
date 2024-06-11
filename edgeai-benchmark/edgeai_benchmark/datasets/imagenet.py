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
import cv2
import pickle
from colorama import Fore
from .. import utils
from .image_cls import *


class BaseImageNetCls(ImageClassification):
    """
    ImageNet Dataset. URL: http://image-net.org

    ImageNet Large Scale Visual Recognition Challenge.
    Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
    Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution)
    International Journal of Computer Vision, 2015.
    """
    def __init__(self, *args, download=False, num_frames=None, name='imagenet', **kwargs):
        self.class_names_dict = None
        self.class_ids_dict = None
        super().__init__(*args, download=download, num_frames=num_frames, name=name, **kwargs)
        assert name is not None, 'Please provide a name for this dataset'
        # create the dataset info
        path = self.kwargs['path']
        root = self._get_root(path)
        download_root = os.path.join(root, 'download')
        extra_path = os.path.join(download_root, 'rawdata_extra')
        synset_words_file = os.path.join(extra_path, 'synset_words.txt')
        if os.path.exists(synset_words_file):
            self.class_names_dict = {}
            with open(synset_words_file) as fp:
                for line in fp:
                    line = line.rstrip()
                    words = line.split(' ')
                    key = words[0]
                    value = ' '.join(words[1:])
                    self.class_names_dict.update({key:value})
                #
            #
            self.class_names_dict = {k:self.class_names_dict[k] for k in sorted(self.class_names_dict.keys())}
            self.class_ids_dict = {k:id for id,(k,v) in enumerate(self.class_names_dict.items())}
            categories = [{'id':i, 'name':v, 'wnid':k} for i, (k,v) in enumerate(self.class_names_dict.items())]
            info = dict(description='ImageNet1K (ILSVRC 2012)', url='https://www.image-net.org/', version='1.0',
                        year='2012',
                        contributor='ImageNet Research Team: Prof. Li Fei-Fei, PI, Stanford University; Prof. Jia Deng, '
                                    'Princeton University; Prof. Olga Russakovsky, Princeton University; Prof. Alex Berg, '
                                    'UNC Chapel Hill, Facebook, Shopagon; Prof. Kai Li, Princeton University ',
                        date_created='2012/june/12')
            self.dataset_store = dict(info=info, categories=categories)
        else:
            self.dataset_store = None
        #
        self.kwargs['dataset_info'] = self.get_dataset_info()

    def get_notice(self):
        notice = f'{Fore.YELLOW}' \
                 f'\nImageNet Dataset: ' \
                 f'\n    ImageNet Large Scale Visual Recognition Challenge, ' \
                 f'\n        Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, ' \
                 f'\n        Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg ' \
                 f'\n        and Li Fei-Fei. (* = equal contribution), IJCV, 2015, ' \
                 f'\n        https://arxiv.org/abs/1409.0575 ' \
                 f'\n    Visit the following url to know more about ImageNet dataset ' \
                 f'\n        http://image-net.org ' \
                  f'\n       and also to register and obtain the download links. '  \
                 f'{Fore.RESET}\n'
        return notice

    def download(self, path, split_file):
        print(utils.log_color('\nINFO', 'downloading and preparing dataset', path + ' This may take some time.'))
        print(self.get_notice())

    def download_extra(self, path, split_file):
        root = self._get_root(path)
        download_root = os.path.join(root, 'download')
        extra_root = os.path.join(download_root, 'rawdata_extra')

        if (not self.force_download) and os.path.exists(extra_root):
            print(utils.log_color('\nINFO', 'dataset info exists - will reuse', extra_root))
            return extra_root
        #
        
        extra_url = 'http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz'
        extra_path = utils.download_file(extra_url, root=download_root, extract_root=extra_root,
                                         force_download=self.force_download)
        if split_file is not None:
            for f in ['train.txt', 'val.txt', 'test.txt']:
                source_file = os.path.join(extra_path, f)
                dest_file = os.path.join(root, f)
                if os.path.exists(source_file) and not os.path.exists(dest_file):
                    shutil.copy2(source_file, dest_file)
                #
            #
        #
        print(utils.log_color('\nINFO', 'dataset ready', path))
        return extra_path

    def _get_root(self, path):
        path = path.rstrip('/')
        root = os.sep.join(os.path.split(path)[:-1])
        return root

    def get_dataset_info(self):
        if 'dataset_info' in self.kwargs:
            return self.kwargs['dataset_info']
        #
        if self.dataset_store is None:
            return None
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


class ImageNetCls(BaseImageNetCls):
    """
    ImageNet Dataset. URL: http://image-net.org
    "ImageNet Large Scale Visual Recognition Challenge."
    Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
    Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution)
    International Journal of Computer Vision, 2015.
    Download page: http://image-net.org/download-images
    """
    def __init__(self, *args, num_classes=1000, num_frames=None, name=None, **kwargs):
        super().__init__(*args, num_classes=num_classes, num_frames=num_frames, name=name, **kwargs)

    def download(self, path, split_file):
        root = self._get_root(path)
        input_message = f'{Fore.YELLOW}' \
                        f'\nImageNet Dataset: ' \
                        f'\n    ImageNet Large Scale Visual Recognition Challenge.' \
                        f'\n        Olga Russakovsky et.al, International Journal of Computer Vision, 2015.\n' \
                        f'\nPlease visit the url: http://image-net.org ' \
                        f'\n    to know more about ImageNet dataset and understand ' \
                        f'\n    the terms and conditions under which it can be used. \n' \
                        f'\nAfter registering and logging in, click on "Download" and then "2012", ' \
                        f'\n    and copy the URL to download the following file.\n' \
                        f'\nPlease enter the full URL of the file - ' \
                        f'Validation images (all tasks). ILSVRC2012_img_val.tar: ' \
                        f'{Fore.RESET}\n'
        dataset_url = input(input_message)
        download_root = os.path.join(root, 'download')
        dataset_path = utils.download_file(dataset_url, root=download_root, extract_root=path,
                                           force_download=self.force_download)
        extra_path = super().download(path, split_file)
        return [dataset_path, extra_path]


