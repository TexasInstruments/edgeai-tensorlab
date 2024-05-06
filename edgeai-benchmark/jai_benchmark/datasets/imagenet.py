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
        extra_url = 'http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz'
        root = self._get_root(path)
        download_root = os.path.join(root, 'download')
        extra_root = os.path.join(download_root, 'rawdata_extra')
        extra_path = utils.download_file(extra_url, root=download_root, extract_root=extra_root,
                                         force_download=self.force_download)
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
        print(utils.log_color('\nINFO', 'dataset ready', path))
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



###############################################################
# OTHER IMAGENET SUBSETS - FOR EXPERIMENTATION
##############################################################

class TinyImageNet200Cls(BaseImageNetCls):
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
    def __init__(self, *args, num_classes=200, **kwargs):
        super().__init__(*args, num_classes=num_classes, **kwargs)

    def download(self, path, split_file):
        root = self._get_root(path)
        extra_path = super().download(path, split_file=None)
        dataset_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        download_root = os.path.join(root, 'download')
        extract_root = os.path.join(download_root, 'rawdata')
        dataset_path = utils.download_file(dataset_url, root=download_root, extract_root=extract_root,
                                           force_download=self.force_download)
        extract_path = os.path.join(extract_root, 'tiny-imagenet-200/val/images')
        os.makedirs(path, exist_ok=True)
        for f in os.listdir(extract_path):
            shutil.copy2(os.path.join(extract_path,f), os.path.join(path,f))
        #
        with open(os.path.join(extract_root, 'tiny-imagenet-200/val/val_annotations.txt')) as fp:
            lines = [line.split('\t') for line in fp]
            lines = [[line[0], self.class_ids_dict[line[1]]] for line in lines]
            lines = [f'{line[0]} {line[1]}' for line in lines]
        #
        with open(split_file, 'w') as fp:
            fp.write('\n'.join(lines))
        #
        return [dataset_path, extra_path]


class ImageNetResized64x64Cls(BaseImageNetCls):
    """
    ImageNet Dataset. URL: http://image-net.org
    ImageNet Large Scale Visual Recognition Challenge.
    Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
    Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution)
    International Journal of Computer Vision, 2015.
    Download page: http://image-net.org/download-images

    This dataset containes ImageNet resized to a smaller size.
    More Information:
        https://patrykchrabaszcz.github.io/Imagenet32/
        ‘labels’- number representing image class, indexing starts at 1 and it uses mapping from the map_clsloc.txt file provided in original Imagenet devkit
        Link: https://github.com/PatrykChrabaszcz/Imagenet32_Scripts/blob/master/map_clsloc.txt
        Raw link: https://raw.githubusercontent.com/PatrykChrabaszcz/Imagenet32_Scripts/master/map_clsloc.txt
        Note: this seems to be different labels compared to what Caffe/TorchVision and TensorflowDatasets uses
    More Information:
        http://image-net.org/download-images
    More Information:
        https://www.tensorflow.org/datasets/catalog/imagenet_resized
    """
    def __init__(self, *args, num_classes=1000, torch_labels=True, **kwargs):
        self.torch_labels = torch_labels
        super().__init__(*args, num_classes=num_classes, **kwargs)

    def download(self, path, split_file):
        root = self._get_root(path)
        download_root = os.path.join(root, 'download')
        dataset_url = 'http://image-net.org/data/downsample/Imagenet64_val.zip'
        extract_root = os.path.join(download_root, 'rawdata')
        dataset_path = utils.download_file(dataset_url, root=download_root, extract_root=extract_root,
                                           force_download=self.force_download)
        with open(os.path.join(dataset_path, 'val_data'), 'rb') as fp:
            val_data = pickle.load(fp)
            data = val_data['data']
            labels = val_data['labels']
        #
        map_clsloc_id_to_torch_labels = None
        if hasattr(self, 'torch_labels') and self.torch_labels:
            extra_root = os.path.join(download_root, 'rawdata_extra')
            map_clsloc_link = 'https://raw.githubusercontent.com/PatrykChrabaszcz/Imagenet32_Scripts/master/map_clsloc.txt'
            map_clsloc_path = utils.download_file(map_clsloc_link, root=extra_root)
            with open(map_clsloc_path) as fp:
                map_clsloc_lines = [line for line in fp]
                map_clsloc_names = [line.split(' ')[:2] for line in map_clsloc_lines]
                map_clsloc_names_dict = {int(id):name  for name, id in map_clsloc_names}
            #
        #
        lines = []
        os.makedirs(path, exist_ok=True)
        for d_id, (d_img, d_lbl) in enumerate(zip(data,labels)):
            img_basename = f'val_{d_id}.png'
            img_filename = os.path.join(path, img_basename)
            d_img = d_img.reshape((3,64,64)).transpose(1,2,0)
            d_img = d_img[:,:,::-1]
            cv2.imwrite(img_filename, d_img)
            if map_clsloc_id_to_torch_labels is not None:
                class_name = map_clsloc_names_dict[d_lbl]
                d_lbl = self.class_ids_dict[class_name]
            #
            lines.append(f'{img_basename} {d_lbl}')
        #
        with open(split_file, 'w') as fp:
            fp.write('\n'.join(lines))
        #
        return [dataset_path, None]


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
    def __init__(self, *args, num_classes=120, **kwargs):
        super().__init__(*args, num_classes=num_classes, **kwargs)

    def download(self, path, split_file):
        root = self._get_root(path)
        download_root = os.path.join(root, 'download')
        tar_filename = 'ILSVRC2012_img_train_t3.tar'
        full_tar_filename = os.path.join(download_root, tar_filename)
        tmp_extract_root = os.path.join(download_root, 'rawdata')
        if self.force_download or (not os.path.exists(full_tar_filename)):
            input_message = f'\nImageNet Dataset: ' \
                            f'\n    ImageNet Large Scale Visual Recognition Challenge.' \
                            f'\n        Olga Russakovsky et.al, International Journal of Computer Vision, 2015.\n' \
                            f'\nPlease visit the url: http://image-net.org ' \
                            f'\n    to know more about ImageNet dataset and understand ' \
                            f'\n    the terms and conditions under which it can be used. \n' \
                            f'\nAfter registering and logging in, click on "Download" and then "2012", ' \
                            f'\n    and copy the URL to download the following file.\n' \
                            f'\nPlease enter the full URL of the file - ' \
                            f'Training images (Task 3), {tar_filename} : '
            dataset_url = input(input_message)
            dataset_path = utils.download_file(dataset_url, root=download_root, extract_root=tmp_extract_root,
                                               force_download=self.force_download)
        #
        dataset_path = utils.extract_archive(full_tar_filename, tmp_extract_root)

        os.makedirs(path, exist_ok=True)
        self.extract_all(tmp_extract_root, path)
        if not os.path.exists(split_file):
            # we don't want the base class to touch or create the split_file this has its own
            print(utils.log_color('\nINFO',  'split_file exists - will reuse', path))
            extra_path = super().download(path, split_file=None)
            self._create_split(path, split_file)
        else:
            print(self.get_notice())
            extra_path = None
        #
        return [path, extra_path]

    def _create_split(self, path, split_file):
        root = self._get_root(path)
        image_folders = utils.list_dir(path)
        image_names_dict = dict()
        for image_folder in image_folders:
            image_folder_base = os.path.basename(image_folder)
            image_filenames = utils.list_files(image_folder)
            image_filenames = [os.path.join(image_folder_base, os.path.basename(f)) for f in image_filenames]
            image_names_dict[image_folder_base] = image_filenames
        #
        with open(split_file, 'w') as fp:
            for class_name, imglist in image_names_dict.items():
                class_id = self.class_ids_dict[class_name]
                for img in imglist:
                    fp.write(f'{img} {class_id}\n')
                #
            #
        #

    def extract_all(self, src_dir, dst_dir):
        extract_tars = utils.list_files(src_dir)
        for extract_tar in extract_tars:
            category_name = os.path.basename(extract_tar)
            category_name = os.path.splitext(category_name)[0]
            tmp_extract_class_dir = os.path.join(dst_dir, category_name)
            utils.extract_archive(extract_tar, tmp_extract_class_dir, verbose=False)
        #


class ImageNetPseudo120Cls(ImageNetDogs120Cls):
    ''''
    Same as ImageNetDogs120Cls, but expects a valid split file to be provided.
    This dataset with a suitable split file can be used to simulate the
    arracy obtained with the full imagenet dataset.
    '''
    def __init__(self, *args, num_classes=120, **kwargs):
        split_file = kwargs['split']
        assert os.path.exists(split_file), f'{self.__class__.__name__}: file not found - {split_file}'
        super().__init__(*args, num_classes=num_classes, **kwargs)
