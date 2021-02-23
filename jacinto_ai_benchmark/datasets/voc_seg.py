import os
import glob
import random
import numpy as np
import PIL
import cv2
from .. import utils

__all__ = ['VOC2012Segmentation']

class VOC2012Segmentation():
    def __init__(self, num_classes=21, ignore_label=255, **kwargs):
        self.kwargs = kwargs
        assert 'path' in kwargs and 'split' in kwargs, 'path and split must provided'
        self.kwargs['num_frames'] = self.kwargs.get('num_frames', None)

        # mapping for voc 21 class segmentation
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        list_txt = os.path.join(self.kwargs['path'], 'ImageSets', 'Segmentation', self.kwargs['split']+'.txt')
        file_indexes = list(open(list_txt))

        base_path_images = os.path.join(self.kwargs['path'], 'JPEGImages')
        images = [base_path_images + '/' +file_index.rstrip() + '.jpg' for file_index in file_indexes]
        self.imgs = sorted(images)
        #
        base_path_labels = os.path.join(self.kwargs['path'], 'SegmentationClassRaw')
        labels = [base_path_labels + '/' +file_index.rstrip() + '.png' for file_index in file_indexes]
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
            #label_img = self.label_lut[label_img]

            output = predictions[n]
            output = output.astype(np.uint8)
            output = output[0] if (output.ndim > 2 and output.shape[0] == 1) else output
            output = output[:2] if (output.ndim > 2 and output.shape[2] == 1) else output

            cmatrix = utils.confusion_matrix(cmatrix, output, label_img, self.num_classes)
        #
        accuracy = utils.segmentation_accuracy(cmatrix)
        return accuracy

    # def _create_lut(self):
    #     if self.label_dict:
    #         lut = np.zeros(256, dtype=np.uint8)
    #         for k in range(256):
    #             lut[k] = k
    #         for k in self.label_dict.keys():
    #             lut[k] = self.label_dict[k]
    #         return lut
    #     else:
    #         return None
    #     #