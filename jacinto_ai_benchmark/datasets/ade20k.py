import os
import glob
import random
import numpy as np
import PIL
from .. import utils

__all__ = ['ADE20KSegmentation']

class ADE20KSegmentation(utils.ParamsBase):
    def __init__(self, num_classes=151, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        assert 'path' in kwargs and 'split' in kwargs, 'path, split must be provided'
        #assert self.kwargs['split'] in ['training', 'validation']
        self.kwargs['num_frames'] = self.kwargs.get('num_frames', None)
        self.name = "ADE20K"
        self.num_classes = num_classes
        self.load_classes() # mlperf model is trained only for 32 classes

        #self.label_lut = self._create_lut()

        image_dir = os.path.join(self.kwargs['path'], 'images', self.kwargs['split'])
        images_pattern = os.path.join(image_dir, '*.jpg')
        images = glob.glob(images_pattern)
        self.imgs = sorted(images)
        #
        label_dir = os.path.join(self.kwargs['path'], 'annotations', self.kwargs['split'])
        labels_pattern = os.path.join(label_dir, '*.png')
        labels = glob.glob(labels_pattern)
        self.labels = sorted(labels)
        #
        assert len(self.imgs) == len(self.labels), 'length of images must be equal to the length of labels'

        shuffle = self.kwargs['shuffle'] if (isinstance(self.kwargs, dict) and 'shuffle' in self.kwargs) else False
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.imgs)
            random.seed(int(shuffle))
            random.shuffle(self.labels)
        #
        self.num_frames = min(self.kwargs['num_frames'], len(self.imgs)) \
            if (self.kwargs['num_frames'] is not None) else len(self.imgs)
        super().initialize()

    def __getitem__(self, idx, with_label=False):
        if with_label:
            image_file = self.imgs[idx]
            label_file = self.labels[idx]
            return image_file, label_file
        else:
            return self.imgs[idx]
        #

    def __len__(self):
        return self.num_frames

    def __call__(self, predictions, **kwargs):
        return self.evaluate(predictions, **kwargs)

    def evaluate(self, predictions, **kwargs):
        cmatrix = None
        for n in range(self.num_frames):
            image_file, label_file = self.__getitem__(n, with_label=True)
            # image = PIL.Image.open(image_file)
            label_img = PIL.Image.open(label_file)
            label_img = label_img.convert('L')
            label_img = np.array(label_img)
            # label_img = self.label_lut[label_img]

            output = predictions[n]
            output = output.astype(np.uint8)
            output = output[0] if (output.ndim > 2 and output.shape[0] == 1) else output
            output = output[:2] if (output.ndim > 2 and output.shape[2] == 1) else output

            cmatrix = utils.confusion_matrix(cmatrix, output, label_img, self.num_classes)
        #
        accuracy = utils.segmentation_accuracy(cmatrix)
        return accuracy

    # def _create_lut(self):  #reverse class should happen here
    #     if self.label_dict:
    #         lut = np.zeros(256, dtype=np.uint8)
    #         for k in range(256):
    #             lut[k] = k
    #         for k in self.label_dict.keys():
    #             lut[k] = self.label_dict[k]
    #         return lut
    #     else:
    #         return None


    def load_classes(self):
        #ade20k_150_classes_url = "https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/objectInfo150.csv"
        label_dir_txt = os.path.join(self.kwargs['path'], 'objectInfo150.txt')
        with open(label_dir_txt) as f:
            list_ade20k_classes = list(map(lambda x: x.split("\t")[-1], f.read().split("\n")))[1:self.num_classes+1]
            self.classes_reverse = dict(zip([i for i in range(1,self.num_classes+1)],list_ade20k_classes))
            self.classes = dict(zip(list_ade20k_classes,[i for i in range(1,self.num_classes+1)]))