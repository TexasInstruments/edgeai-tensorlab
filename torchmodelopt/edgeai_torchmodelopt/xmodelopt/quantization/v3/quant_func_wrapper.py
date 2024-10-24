from . import quant_func, quant_utils
from ...utils.transformation_utils import wrapped_transformation_fn
from ...utils.hooks import add_example_args_kwargs


def init(module, *args, example_inputs=None, example_kwargs=None, transformation_dict=None, **kwargs):
    add_example_args_kwargs(module, example_inputs=example_inputs, example_kwargs=example_kwargs, transformation_dict=transformation_dict)
    return wrapped_transformation_fn(quant_func.init, module, *args, example_inputs=example_inputs, example_kwargs=example_kwargs, 
                                     transformation_dict=transformation_dict, **kwargs)


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

def remove_loss_branch(*args, **kwargs):
    return wrapped_transformation_fn(quant_utils.remove_loss_branch, *args, **kwargs)

def export(self, *args, transformation_dict = None, is_converted = False, device = 'cpu', make_copy = True, **kwargs):
    if is_converted:
        model = self
    else:
        model = convert(self, transformation_dict = transformation_dict, device = device, make_copy = make_copy)
    model = model.to(device=device)
    model = remove_loss_branch(self, transformation_dict = transformation_dict)
    quant_func.export(model, *args, device = device, make_copy = make_copy, is_converted = True, **kwargs)
    return


def calibrate(*args, **kwargs):
    return quant_func.calibrate(*args, freeze_bn = freeze, **kwargs)


def remove_hooks(*args, **kwargs):
    return wrapped_transformation_fn(quant_func.remove_hooks, *args, **kwargs)


forward = quant_func.forward
load_weights = quant_func.load_weights