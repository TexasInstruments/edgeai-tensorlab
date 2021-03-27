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
from .imagenet import *


class TinyImageNetCls(BaseImageNetCls):
    """
    ImageNet Dataset. URL: http://image-net.org
    ImageNet Large Scale Visual Recognition Challenge.
    Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
    Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution)
    International Journal of Computer Vision, 2015.
    Download page: http://image-net.org/download-images

    Tiny ImageNet
    This is a miniature of ImageNet classification Challenge.
    We thank Fei-Fei Li, Andrej Karpathy, and Justin Johnson for providing this dataset as part of their cs231n course at Stanford university http://cs231n.stanford.edu/
    https://www.kaggle.com/c/tiny-imagenet
    http://cs231n.stanford.edu/
    http://cs231n.stanford.edu/tiny-imagenet-200.zip
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def download(self, path):
        root = self._get_root(path)
        dataset_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        dataset_path = utils.download_file(dataset_url, root=root, extract_root=path)
        extra_path = super().download()
        return [dataset_path, extra_path]


class ImageNetResized64x64Cls(BaseImageNetCls):
    """
    ImageNet Dataset. URL: http://image-net.org
    ImageNet Large Scale Visual Recognition Challenge.
    Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
    Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution)
    International Journal of Computer Vision, 2015.
    Download page: http://image-net.org/download-images

    This dataset contained ImageNet resized to a smaller size.
    http://image-net.org/download-images
    https://www.tensorflow.org/datasets/catalog/imagenet_resized
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def download(self, path):
        root = self._get_root(path)
        dataset_url = 'http://image-net.org/data/downsample/Imagenet64_val.zip'
        dataset_path = utils.download_file(dataset_url, root=root, extract_root=path)
        extra_path = super().download()
        return [dataset_path, extra_path]


class ImageNetDogsCls(BaseImageNetCls):
    """
    ImageNet Dataset. URL: http://image-net.org
    ImageNet Large Scale Visual Recognition Challenge.
    Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
    Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution)
    International Journal of Computer Vision, 2015.
    Download page: http://image-net.org/download-images

    Fine-grained classification on 100+ dog categories.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def download(self, path):
        root = self._get_root(path)
        input_message = '\nplease input the full URL of the following file from ' \
                        '\nhttp://image-net.org/download-images in the page 2012: ' \
                        '\nTraining images (Task 3), ILSVRC2012_img_train_t3.tar : '
        dataset_url = input(input_message)
        dataset_path = utils.download_file(dataset_url, root=root, extract_root=path)
        extra_path = super().download()
        return [dataset_path, extra_path]
