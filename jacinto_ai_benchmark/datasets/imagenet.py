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

"""
Reference:

Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy,
Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution)
ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision, 2015.
http://www.image-net.org/
"""


import os
from .. import utils
from .image_cls import *


class ImageNetCls(ImageCls):
    """
    ImageNet Dataset http://image-net.org
    Citation:
    ImageNet Large Scale Visual Recognition Challenge.
    Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
    Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution)
    International Journal of Computer Vision, 2015.
    """
    def __init__(self, *args, download=False, **kwargs):
        assert 'path' in kwargs, 'path must be provided in kwargs'
        path = kwargs['path']
        if download:
            self.download(path)
        #
        super().__init__(*args, **kwargs)
        root = self._get_root(path)
        synset_words_file = os.path.join(root, 'synset_words.txt')
        if os.path.exists(synset_words_file):
            self.class_mapping = {}
            with open(synset_words_file) as fp:
                for line in fp:
                    line = line.rstrip()
                    words = line.split(' ')
                    key = words[0]
                    value = ' '.join(words[1:])
                    self.class_mapping.update({key:value})
                #
            #
            self.class_names = [k for k,v in self.class_mapping.items()]
            self.class_descriptions = [v for k,v in self.class_mapping.items()]
        else:
            self.class_mapping = None
            self.class_names = None
            self.class_descriptions  = None
        #

    def download(self, path):
        root = self._get_root(path)
        if os.path.exists(path):
            return
        #
        print('Important: Please visit the urls: http://image-net.org/ http://image-net.org/about-overview and '
              'http://image-net.org/download-faq to understand more about ImageNet dataset '
              'and accept the terms and conditions under which it can be used. '
              'Also, register/signup on that website, request and get permission to download this dataset.')

        dataset_url = 'http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_val.tar'
        extra_url = 'http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz'
        dataset_path = utils.download_file(dataset_url, root=root, extract_root=path)
        extra_path = utils.download_file(extra_url, root=root)
        return

    def _get_root(self, path):
        path = path.rstrip('/')
        root = os.sep.join(os.path.split(path)[:-1])
        return root

