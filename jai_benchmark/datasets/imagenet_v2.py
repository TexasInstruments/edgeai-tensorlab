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
from .image_cls import  ImageCls
from .. import utils


'''
Reference: 
Do ImageNet Classifiers Generalize to ImageNet?
Benjamin Recht∗, Rebecca Roelofs, Ludwig Schmidt, Vaishaal Shankar
https://arxiv.org/abs/1902.10811
Source Code: https://github.com/modestyachts/ImageNetV2'
Download: http://imagenetv2public.s3-website-us-west-2.amazonaws.com/
'''


class ImageNetV2(ImageCls):
    def __init__(self, *args, url=None, download=False, **kwargs):
        self.url = url
        self.class_names_dict = None
        self.class_ids_dict = None
        super().__init__(*args, download=download, **kwargs)

    def get_notice(self):
        notice = '\nThe ImageNetV2 dataset contains new test data for the ImageNet benchmark.' \
                 '\nImageNetV2 Reference: http://people.csail.mit.edu/ludwigs/papers/imagenet.pdf' \
                 '\nImageNetV2 Source Code: https://github.com/modestyachts/ImageNetV2' \
                 '\nOriginal ImageNet Dataset, URL: http://image-net.org'
        return notice

    def download(self, path, split_file):
        print(self.get_notice())
        root = self._get_root(path)
        extract_root = os.path.join(root, 'rawdata')
        extract_path = utils.download_file(self.url, root, extract_root=extract_root)
        return

    def _get_root(self, path):
        path = path.rstrip('/')
        root = os.sep.join(os.path.split(path)[:-1])
        return root


class ImageNetV2A(ImageNetV2):
    def __init__(self, *args, **kwargs):
        url = 'https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-threshold0.7.tar.gz'
        super().__init__(*args, url=url, **kwargs)


class ImageNetV2B(ImageNetV2):
    def __init__(self, *args, **kwargs):
        url = 'https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz'
        super().__init__(*args, url=url, **kwargs)


class ImageNetV2C(ImageNetV2):
    def __init__(self, *args, **kwargs):
        url = 'https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-top-images.tar.gz'
        super().__init__(*args, url=url, **kwargs)
