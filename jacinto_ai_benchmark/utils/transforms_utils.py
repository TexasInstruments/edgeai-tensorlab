from .attr_base import *


class TransformsCompose(AttrBase):
    def __init__(self, transforms, **kwargs):
        self.transforms = transforms
        self.kwargs = kwargs

    def __call__(self, tensor, info_dict):
        for t in self.transforms:
            tensor, info_dict = t(tensor, info_dict)
        #
        return tensor, info_dict

    def append(self, t):
        self.transforms.append(t)




