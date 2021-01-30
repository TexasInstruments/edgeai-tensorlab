import random
import numpy as np
import PIL
from .. import utils


class ImageClassification():
    def __init__(self, dest_dir=None, label_offset=0, **kwargs):
        self.kwargs = kwargs
        self.imgs = utils.get_data_list(input=kwargs, dest_dir=dest_dir)
        self.num_frames = kwargs.get('num_frames',len(self.imgs))
        self.label_offset = label_offset
        self.label_offset_for_gt = True
        if self.label_offset_for_gt:
            for line_idx in range(len(self.imgs)):
                line = self.imgs[line_idx]
                words = line.split(' ')
                if len(words)>0:
                    words[1] = str(int(words[1]) + self.label_offset)
                    line = ' '.join(words)
                    self.imgs[line_idx] = line
                #
            #
        #
        shuffle = kwargs.get('shuffle', False)
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.imgs)
        #

    def __getitem__(self, idx):
        with_label = self.kwargs.get('with_label', False)
        words = self.imgs[idx].split(' ')
        image_name = words[0]
        if with_label:
            assert len(words)>0, f'ground truth requested, but missing at the dataset entry for {words}'
            label = int(words[1])
            return image_name, label
        else:
            return image_name
        #

    def __len__(self):
        return self.num_frames

    def __call__(self, predictions):
        return self.evaluate(predictions)

    def get_imgs(self):
        return self.imgs

    def evaluate(self, predictions):
        metric_tracker = utils.AverageMeter(name='Classification Accuracy Top1%')
        in_lines = self.imgs
        for n in range(self.num_frames):
            words = in_lines[n].split(' ')
            label = int(words[1])
            accuracy = self.classification_accuracy(predictions[n], label)
            metric_tracker.update(accuracy)
        #
        return {metric_tracker.name:metric_tracker.avg}

    def classification_accuracy(self, output, target, multiplier=100.0):
        if not self.label_offset_for_gt:
            output = output - self.label_offset
        #
        accuracy = 1.0 if (output == target) else 0.0
        accuracy = accuracy * multiplier
        return accuracy

