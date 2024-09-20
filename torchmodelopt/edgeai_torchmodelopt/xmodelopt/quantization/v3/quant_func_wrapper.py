from copy import deepcopy
import functools

from . import quant_func
from .quant_func import ModelQuantFormat
from ...utils import TransformationWrapper,apply_tranformation_to_submodules



def wrapped_transformation_fn(fn, model, *args, transformation_dict=None, **kwargs):
    if transformation_dict is not None:
        transformation_dict = deepcopy(transformation_dict)
        for key, value in transformation_dict.items():
            if value is None:
                transformation_dict[key] = fn
            elif isinstance(value, TransformationWrapper) and value.fn is None:
                value.fn = fn
        return apply_tranformation_to_submodules(model,transformation_dict, *args, **kwargs)
    else:
        return fn(model, *args, **kwargs)


def init(*args, **kwargs):
    return wrapped_transformation_fn(quant_func.init, *args, **kwargs)


def train(*args, **kwargs):
    return wrapped_transformation_fn(quant_func.train, *args, **kwargs)


def freeze(*args, **kwargs):
    return wrapped_transformation_fn(quant_func.freeze, *args, **kwargs)


def unfreeze(*args, **kwargs):
    return wrapped_transformation_fn(quant_func.unfreeze, *args, **kwargs)


def convert(*args, **kwargs):
    return wrapped_transformation_fn(quant_func.convert, *args, **kwargs)

def insert_all_hooks(*args, **kwargs):
    return wrapped_transformation_fn(quant_func.insert_all_hooks, *args, **kwargs)

insert_all_hooks = functools.partial(wrapped_transformation_fn, fn=quant_func.insert_all_hooks)

def export(self, *args, transformation_dict = None, is_converted = False, device='cpu', make_copy=True, **kwargs):
    if is_converted:
        model = self
    else:
        model = convert(self, transformation_dict=transformation_dict, device=device, make_copy=make_copy)
    return quant_func.export(model, *args, device=device, make_copy=make_copy, is_converted=True, **kwargs)

def calliberate(*args, **kwargs):
    return quant_func.calibrate(*args, freeze_bn= freeze, **kwargs)

remove_hooks = quant_func.remove_hooks
forward = quant_func.forward
load_weights = quant_func.load_weights