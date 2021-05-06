import os
import random
from .. import utils

class ImagePixel2Pixel(utils.ParamsBase):
    def __init__(self, download=False, dest_dir=None, **kwargs):
        super().__init__()
        self.force_download = True if download == 'always' else False
        assert 'path' in kwargs and 'split' in kwargs, 'path and split must be provided in kwargs'
        path = kwargs['path']
        split_file = kwargs['split']
        # download the data if needed
        if download:
            if (not self.force_download) and os.path.exists(path):
                print(utils.log_color('\nINFO', 'dataset exists - will reuse', path))
            else:
                self.download(path, split_file)
            #
        #
        assert os.path.exists(path) and os.path.isdir(path), \
            utils.log_color('\nERROR', 'dataset path is empty', path)
        self.kwargs = kwargs

        # create list of images and classes
        path = kwargs.get('path','')
        list_file = kwargs['split']
        with open(list_file) as list_fp:
            in_files = [row.rstrip().split(' ') for row in list_fp]
            in_files = [(os.path.join(path, f[0]), os.path.join(path, f[1])) for f in in_files]
            self.imgs = in_files
        #

        self.num_frames = kwargs.get('num_frames',len(self.imgs))
        shuffle = kwargs.get('shuffle', False)
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.imgs)
        #
        # call the utils.ParamsBase.initialize()
        super().initialize()

    def download(self, path, split_file):
        return None

    def __getitem__(self, index, **kwargs):
        with_label = kwargs.get('with_label', False)
        words = self.imgs[index]
        image_name = words[0]
        if with_label:
            assert len(words)>0, f'ground truth requested, but missing at the dataset entry for {words}'
            label_name = words[1]
            return image_name, label_name
        else:
            return image_name
        #

    def __len__(self):
        return self.num_frames
