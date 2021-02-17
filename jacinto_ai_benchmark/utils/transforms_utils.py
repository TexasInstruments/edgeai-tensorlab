from .misc_utils import *


class TransformsCompose:
    def __init__(self, transforms, **kwargs):
        self.transforms = transforms
        # the keyword arguments passed to this class are set as attributes
        # so that they are easily accessible from outside
        for k, v in kwargs.items():
            setattr(self, k, v)
        #

    def __call__(self, tensor, info_dict):
        for t in self.transforms:
            tensor, info_dict = t(tensor, info_dict)
        #
        return tensor, info_dict

    def append(self, t):
        self.transforms.append(t)

    def get_param(self, param_name):
        return self.__dict__[param_name]


