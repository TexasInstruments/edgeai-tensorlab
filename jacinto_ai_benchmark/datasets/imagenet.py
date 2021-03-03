import os
from .. import utils
from .image_cls import *

''' ImageNet Synset description from http://image-net.org
    Citation:
    ImageNet Large Scale Visual Recognition Challenge.
    Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang,
    Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution)
    International Journal of Computer Vision, 2015.
'''
class ImageNetCls(ImageCls):
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

