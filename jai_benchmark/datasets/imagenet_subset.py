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

    def download(self, path, split_file):
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

    def download(self, path, split_file):
        root = self._get_root(path)
        dataset_url = 'http://image-net.org/data/downsample/Imagenet64_val.zip'
        dataset_path = utils.download_file(dataset_url, root=root, extract_root=path)
        extra_path = super().download()
        return [dataset_path, extra_path]


class ImageNetDogs120Cls(BaseImageNetCls):
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

    def download(self, path, split_file):
        root = self._get_root(path)
        # we don't want the base class to touch or create the split_file this has its own
        extra_path = super().download(path, split_file=None)
        tar_filename = 'ILSVRC2012_img_train_t3.tar'
        full_tar_filename = os.path.join(root, tar_filename)
        if not os.path.exists(full_tar_filename):
            input_message = \
                        f'\nPlease register/signup on that website http://image-net.org and get the URL to download this dataset.' \
                        f'\nIn the download section, click on the link that says 2012, and copy the URL to download the following file.' \
                        f'\nPlease enter the full URL of the file - ' \
                        f'Training images (Task 3), {tar_filename} : '
            dataset_url = input(input_message)
            dataset_path = utils.download_file(dataset_url, root=root, extract_root=path)
        else:
            tmp_extract_root = os.path.join(root, 'rawdata')
            if not os.path.exists(tmp_extract_root):
                dataset_path = utils.extract_archive(full_tar_filename, tmp_extract_root)
            #
            extract_tars = utils.list_files(tmp_extract_root)
            split = path.split(os.sep)[-1]
            for extract_tar in extract_tars:
                category_name = os.path.basename(extract_tar)
                category_name = os.path.splitext(category_name)[0]
                tmp_extract_class_dir = os.path.join(root, split, category_name)
                utils.extract_archive(extract_tar, tmp_extract_class_dir, verbose=False)
            #
            self._create_split(path, split_file)
        #
        return [path, extra_path]

    def _create_split(self, path, split_file, balanced_sampling=True):
        root = self._get_root(path)
        if os.path.exists(split_file):
            print(f'{Fore.CYAN}INFO:{Fore.YELLOW} split_file exists - will reuse:{Fore.RESET} {path}')
            return split_file
        #
        image_folders = utils.list_dir(path)
        image_names_dict = dict()
        for image_folder in image_folders:
            image_folder_base = os.path.basename(image_folder)
            image_filenames = utils.list_files(image_folder)
            image_filenames = [os.path.join(image_folder_base, os.path.basename(f)) for f in image_filenames]
            image_names_dict[image_folder_base] = image_filenames
        #
        if balanced_sampling:
            sample_size_dict = {k:len(v) for k,v in image_names_dict.items()}
            min_sample_size = min(sample_size_dict.values())
            image_names_dict = {k:v[:min_sample_size] for k,v in image_names_dict.items()}
        #
        with open(split_file, 'w') as fp:
            for class_name, imglist in image_names_dict.items():
                class_id = self.class_ids_dict[class_name]
                for img in imglist:
                    fp.write(f'{img} {class_id}\n')
                #
            #
        #
