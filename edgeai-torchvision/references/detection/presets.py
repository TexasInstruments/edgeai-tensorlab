import os
import torch

basename = os.path.splitext(os.path.basename(__file__))[0]
if __name__.startswith(basename):
    import transforms as T
    import transforms_mosaic as Tm
else:
    from . import transforms as T
    from . import transforms_mosaic as Tm
#

class DetectionPresetTrain:
    def __init__(self, data_augmentation, hflip_prob=0.5, image_mean=(123., 117., 104.), image_size=(512,512)):
        data_augmentation = data_augmentation or 'ssdlite'
        if data_augmentation == 'hflip':
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(p=hflip_prob),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
            ])
        elif data_augmentation == 'ssd':
            self.transforms = T.Compose([
                T.RandomPhotometricDistort(),
                T.RandomZoomOut(fill=list(image_mean)),
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
            ])
        elif data_augmentation == 'ssdlite':
            self.transforms = T.Compose([
                T.RandomIoUCrop(),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
            ])
        elif data_augmentation == 'ssd_fixed_size': #resizes image to fixed resolution
            self.transforms = T.Compose([
                T.RandomPhotometricDistort(),
                T.Expand(),
                T.RandomIoUCrop(),
                T.Resize(op_width=image_size[1], op_height=image_size[0]),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
            ])
        elif data_augmentation == 'mosaic':
            self.transforms = T.Compose([
                T.RandomPhotometricDistort(),
                T.RandomHorizontalFlip(p=hflip_prob),
                T.ToTensor(),
            ])
            self.transforms = Tm.RandomMosaic(self.transforms)
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self):
        self.transforms = T.ToTensor()

    def __call__(self, img, target):
        return self.transforms(img, target)
