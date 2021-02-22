import random
from .. import utils


class ImageCls(utils.ParamsBase):
    def __init__(self, dest_dir=None, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.imgs = utils.get_data_list(input=kwargs, dest_dir=dest_dir)
        self.num_frames = kwargs.get('num_frames',len(self.imgs))
        shuffle = kwargs.get('shuffle', False)
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(self.imgs)
        #
        super().initialize()

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

    def __call__(self, predictions, **kwargs):
        return self.evaluate(predictions, **kwargs)

    def evaluate(self, predictions, **kwargs):
        metric_tracker = utils.AverageMeter(name='accuracy-top1%')
        in_lines = self.imgs
        for n in range(self.num_frames):
            words = in_lines[n].split(' ')
            gt_label = int(words[1])
            accuracy = self.classification_accuracy(predictions[n], gt_label, **kwargs)
            metric_tracker.update(accuracy)
        #
        return {metric_tracker.name:metric_tracker.avg}

    def classification_accuracy(self, prediction, target, label_offset_pred=0, label_offset_gt=0,
                                multiplier=100.0, **kwargs):
        prediction = prediction + label_offset_pred
        target = target + label_offset_gt
        accuracy = 1.0 if (prediction == target) else 0.0
        accuracy = accuracy * multiplier
        return accuracy

