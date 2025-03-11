from . import quant_func
from ...utils.transformation_utils import wrapped_transformation_fn
from ... import utils


def init(module, *args, example_inputs=None, example_kwargs=None, transformation_dict=None, **kwargs):
    example_kwargs = example_kwargs or {}
    utils.add_example_args_kwargs(module,example_inputs=example_inputs, example_kwargs=example_kwargs,transformation_dict=transformation_dict)
    return wrapped_transformation_fn(quant_func.init, module, *args, example_inputs=example_inputs, example_kwargs=example_kwargs, transformation_dict=transformation_dict,**kwargs)


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


def export(self, *args, transformation_dict=None, is_converted=False, device='cpu', make_copy=True, **kwargs):
    if is_converted:
        model = self
    else:
        model = convert(self, transformation_dict=transformation_dict, device=device, make_copy=make_copy)
    return quant_func.export(model, *args, device=device, make_copy=make_copy, is_converted=True, **kwargs)


def calibrate(*args, **kwargs):
    return quant_func.calibrate(*args, freeze_bn= freeze, **kwargs)


def remove_hooks(*args, **kwargs):
    return wrapped_transformation_fn(quant_func.remove_hooks, *args, **kwargs)


forward = quant_func.forward
load_weights = quant_func.load_weights
