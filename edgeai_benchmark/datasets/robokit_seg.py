import numpy as np
import PIL

from .. import utils
from .image_seg import ImageSegmentation


class RobokitSegmentation(ImageSegmentation):
    def __init__(self, download=False, dest_dir=None, num_frames=None, name='RobokitSegmentation', **kwargs):
        super().__init__(download=download, dest_dir=dest_dir, num_frames=num_frames, name=name, **kwargs)
        assert 'path' in self.kwargs and 'split' in self.kwargs, 'path and split must be provided in kwargs'
        assert 'num_classes' in kwargs, f'num_classes must be provided while creating {self.__class__.__name__}'
        assert name is not None, 'Please provide a name for this dataset'
        self.num_classes = self.kwargs['num_classes']

    def download(self, path, split):
        root = path
        print(utils.log_color('\nINFO', 'downloading and preparing dataset', path + ' This may take some time.'))
        dataset_url = 'https://software-dl.ti.com/jacinto7/esd/modelzoo/08_06_00_01/datasets/ti-robokit_semseg_zed1hd.zip'
        download_root = os.path.join(root, 'download')
        dataset_path = utils.download_file(dataset_url, root=download_root, extract_root=root)
        print(utils.log_color('\nINFO', 'dataset ready', path))
        return

