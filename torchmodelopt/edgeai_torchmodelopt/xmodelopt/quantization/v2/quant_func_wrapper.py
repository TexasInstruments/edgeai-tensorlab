import functools

from . import quant_func
from .quant_func import ModelQuantFormat
from ...utils.transformation_utils import wrapped_transformation_fn



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


def export(self, *args, transformation_dict=None, is_converted=False, device='cpu', make_copy=True, **kwargs):
    if is_converted:
        model = self
    else:
        model = convert(self, transformation_dict=transformation_dict, device=device, make_copy=make_copy)
    return quant_func.export(model, *args, device=device, make_copy=make_copy, is_converted=True, **kwargs)


def calibrate(*args, **kwargs):
    return quant_func.calibrate(*args, freeze_bn= freeze, **kwargs)


remove_hooks = quant_func.remove_hooks
forward = quant_func.forward
load_weights = quant_func.load_weights
