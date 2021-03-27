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
from .. import utils
from .image_cls import *


class BaseImageNetCls(ImageCls):
    """
    ImageNet Dataset. URL: http://image-net.org

    ImageNet Large Scale Visual Recognition Challenge.
    Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
    Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution)
    International Journal of Computer Vision, 2015.
    """
    def __init__(self, *args, download=False, **kwargs):
        self.class_names_dict = None
        self.class_ids_dict = None
        super().__init__(*args, download=download, **kwargs)

    def get_notice(self):
        notice = '\nImageNet Dataset, URL: http://image-net.org' \
                 '\nIMPORTANT: Please visit the urls: http://image-net.org/ http://image-net.org/about-overview and ' \
                 '\nhttp://image-net.org/download-faq to understand more about ImageNet dataset ' \
                 '\nand the terms and conditions under which it can be used. '
        return notice

    def download(self, path, split_file):
        print(self.get_notice())
        extra_url = 'http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz'
        root = self._get_root(path)
        extra_root = os.path.join(root, 'extra')
        extra_path = utils.download_file(extra_url, root=extra_root)
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
        #
        if split_file is not None:
            if not os.path.exists(split_file):
                for f in ['train.txt', 'val.txt', 'test.txt']:
                    shutil.copy2(os.path.join(extra_path, f), os.path.join(root, f))
                #
            #
        #
        return extra_path

    def _get_root(self, path):
        path = path.rstrip('/')
        root = os.sep.join(os.path.split(path)[:-1])
        return root


class ImageNetCls(BaseImageNetCls):
    """
    ImageNet Dataset. URL: http://image-net.org
    "ImageNet Large Scale Visual Recognition Challenge."
    Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
    Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution)
    International Journal of Computer Vision, 2015.
    Download page: http://image-net.org/download-images
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def download(self, path, split_file):
        root = self._get_root(path)
        input_message = \
                    f'\nPlease register/signup on that website http://image-net.org and get the URL to download this dataset.' \
                    f'\nIn the download section, click on the link that says 2012, and copy the URL to download the following file.' \
                    f'\nPlease enter the full URL of the file - ' \
                    f'\nValidation images (all tasks). ILSVRC2012_img_val.tar: '
        dataset_url = input(input_message)
        dataset_path = utils.download_file(dataset_url, root=root, extract_root=path)
        extra_path = super().download(path, split_file)
        return [dataset_path, extra_path]

