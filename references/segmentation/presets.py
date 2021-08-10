import transforms as T
import transforms_mosaic as Tm


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5,
                 mean=(123.675, 116.28, 103.53), scale=(0.017125, 0.017507, 0.017429),
                 data_augmentation=None):
        min_ratio = 0.5
        max_ratio = 2.0
        list_size = isinstance(base_size, (list,tuple))
        if list_size:
            min_size = (int(min_ratio * base_size[0]), int(min_ratio * base_size[1]))
            max_size = (max_ratio * base_size[0], max_ratio * base_size[1])
        else:
            min_size = int(min_ratio * base_size)
            max_size = int(max_ratio * base_size)
        #

        # Note: input is divided by 255 before this mean/std is applied
        # Note: can potentially use direct_float in ToTensor and then T.NormalizeMeanScale() to avoid division by 255
        float_mean = [m/255.0 for m in mean]
        float_std = [(1.0/s)/255.0 for s in scale]

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=float_mean, std=float_std),
        ])
        self.transforms = T.Compose(trans)

        if data_augmentation == 'mosaic':
            self.transforms = Tm.RandomMosaic(self.transforms)
        #

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(123.675, 116.28, 103.53), scale=(0.017125, 0.017507, 0.017429)):
        # Note: input is divided by 255 before this mean/std is applied
        # Note: can potentially use direct_float in ToTensor and then T.NormalizeMeanScale() to avoid division by 255
        float_mean = [m/255.0 for m in mean]
        float_std = [(1.0/s)/255.0 for s in scale]
		
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=float_mean, std=float_std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)
