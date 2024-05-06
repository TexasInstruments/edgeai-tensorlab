import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode


class ClassificationPresetTrain:
    def __init__(self, crop_size, image_mean=(123.675, 116.28, 103.53), image_scale=(0.017125, 0.017507, 0.017429), hflip_prob=0.5,
                 auto_augment_policy=None, random_erase_prob=0.0):
        # Note: input is divided by 255 before this mean/std is applied
        # Note: can potentially use direct_float in ToTensor and then T.NormalizeMeanScale() to avoid division by 255
        float_mean = [m/255.0 for m in image_mean]
        float_std = [(1.0/s)/255.0 for s in image_scale]
        
        trans = [transforms.RandomResizedCrop(crop_size)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                trans.append(autoaugment.RandAugment())
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide())
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy))
        trans.extend([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=float_mean, std=float_std),
        ])
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(self, crop_size, resize_size=256, image_mean=(123.675, 116.28, 103.53), image_scale=(0.017125, 0.017507, 0.017429),
                 interpolation=InterpolationMode.BILINEAR):
        # Note: input is divided by 255 before this mean/std is applied
        # Note: can potentially use direct_float in ToTensor and then T.NormalizeMeanScale() to avoid division by 255
        float_mean = [m/255.0 for m in image_mean]
        float_std = [(1.0/s)/255.0 for s in image_scale]

        self.transforms = transforms.Compose([
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=float_mean, std=float_std),
        ])

    def __call__(self, img):
        return self.transforms(img)
