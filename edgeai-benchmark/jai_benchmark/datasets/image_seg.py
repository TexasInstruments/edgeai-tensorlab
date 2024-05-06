import numpy as np
import PIL

from .. import utils
from .image_pix2pix import ImagePixel2Pixel


class ImageSegmentation(ImagePixel2Pixel):
    def __init__(self, download=False, dest_dir=None, num_frames=None, name=None, **kwargs):
        super().__init__(download=download, dest_dir=dest_dir, num_frames=num_frames, name=name, **kwargs)
        assert 'path' in self.kwargs and 'split' in self.kwargs, 'path and split must be provided in kwargs'
        assert 'num_classes' in kwargs, f'num_classes must be provided while creating {self.__class__.__name__}'
        assert name is not None, 'Please provide a name for this dataset'
        self.num_classes = self.kwargs['num_classes']

    def __call__(self, predictions, **kwargs):
        return self.evaluate(predictions, **kwargs)

    def evaluate(self, predictions, **kwargs):
        cmatrix = None
        for n in range(self.num_frames):
            image_file, label_file = self.__getitem__(n, with_label=True)
            label_img = PIL.Image.open(label_file)
            # reshape prediction is needed
            output = predictions[n]
            output = output.astype(np.uint8)
            output = output[0] if (output.ndim > 2 and output.shape[0] == 1) else output
            output = output[:2] if (output.ndim > 2 and output.shape[2] == 1) else output
            # compute metric
            cmatrix = utils.confusion_matrix(cmatrix, output, label_img, self.num_classes)
        #
        accuracy = utils.segmentation_accuracy(cmatrix)
        return accuracy

