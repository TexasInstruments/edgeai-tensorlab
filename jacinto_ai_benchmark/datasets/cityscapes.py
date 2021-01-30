import os
import glob
import random
import numpy as np
import PIL
import cv2
from .. import utils

__all__ = ['CityscapesSegmentation']

class CityscapesSegmentation():
    def __init__(self, inData, num_imgs=None):
        assert isinstance(inData, dict) and 'path' in list(inData.keys()) and 'split' in list(inData.keys()), \
            'inData must be a dict'
        #

        # mapping for cityscapes 19 class segmentation
        self.num_classes = 19
        self.label_dict = {0:255, 1:255, 2:255, 3:255, 4:255, 5:255, 6:255, 7:0, 8:1, 9:255,
                     10:255, 11:2, 12:3, 13:4, 14:255, 15:255, 16:255, 17:5, 18:255, 19:6,
                     20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:255,
                     30:255, 31:16, 32:17, 33:18, 255:255}
        self.label_lut = self._create_lut()

        image_dir = os.path.join(inData['path'], 'leftImg8bit', inData['split'])
        images_pattern = os.path.join(image_dir, '*', '*.png')
        images = glob.glob(images_pattern)
        self.imgs = sorted(images)
        #
        label_dir = os.path.join(inData['path'], 'gtFine', inData['split'])
        labels_pattern = os.path.join(label_dir, '*', '*_labelIds.png')
        labels = glob.glob(labels_pattern)
        self.labels = sorted(labels)
        #
        assert len(self.imgs) == len(self.labels), 'length of images must be equal to the length of labels'

        shuffle = inData['shuffle'] if (isinstance(inData, dict) and 'shuffle' in inData) else False
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.imgs)
            random.seed(int(shuffle))
            random.shuffle(self.labels)
        #
        self.num_imgs = min(num_imgs, len(self.imgs)) if (num_imgs is not None) else len(self.imgs)

    def __getitem__(self, idx, with_label=False):
        if with_label:
            image_file = self.imgs[idx]
            image = PIL.Image.open(image_file)
            label_file = self.labels[idx]
            label_img = PIL.Image.open(label_file)
            label_img = label_img.convert('L')
            label_img = np.array(label_img)
            label_img = self.label_lut[label_img]
            return image, label_img
        else:
            return self.imgs[idx]
        #

    def __len__(self):
        return self.num_imgs

    def __call__(self, predictions):
        return self.evaluate(predictions)

    def get_imgs(self):
        return self.imgs

    def _create_lut(self):
        if self.label_dict:
            lut = np.zeros(256, dtype=np.uint8)
            for k in range(256):
                lut[k] = k
            for k in self.label_dict.keys():
                lut[k] = self.label_dict[k]
            return lut
        else:
            return None
        #

    def evaluate(self, predictions):
        cmatrix = None
        for n in range(self.num_imgs):
            image, label_img = self.__getitem__(n, with_label=True)
            gtHeight, gtWidth = label_img.shape[:2]

            output = predictions[n]
            output = output.astype(np.uint8)
            output = output[0] if (output.ndim > 2 and output.shape[0] == 1) else output
            output = output[:2] if (output.ndim > 2 and output.shape[2] == 1) else output

            resample_type = cv2.INTER_NEAREST if isinstance(output, np.ndarray) else PIL.Image.NEAREST
            output = utils.resize_pad_crop_image(output, resize_w=gtWidth, resize_h=gtHeight, inResizeType=0,
                                                 resample_type=resample_type)
            cmatrix = utils.confusion_matrix(cmatrix, output, label_img, self.num_classes)
        #

        accuracy = utils.segmentation_accuracy(cmatrix)
        return accuracy


