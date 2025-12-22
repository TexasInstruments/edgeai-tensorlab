from ...utils.transformation_utils import wrapped_transformation_fn
from ... import utils
from . import sparsity_func


def init(module, *args, example_inputs, example_kwargs=None, transformation_dict=None, **kwargs):
    example_kwargs = example_kwargs or {}
    utils.add_example_args_kwargs(module, example_inputs=example_inputs, example_kwargs=example_kwargs, transformation_dict=transformation_dict)
    return wrapped_transformation_fn(sparsity_func.init, module, *args, example_inputs=example_inputs, example_kwargs=example_kwargs, transformation_dict=transformation_dict, **kwargs)


def calculate_sparsity(*args, **kwargs):
    return wrapped_transformation_fn(sparsity_func.calculate_sparsity, *args, **kwargs)


def train(*args, **kwargs):
    return wrapped_transformation_fn(sparsity_func.train, *args, **kwargs)


def eval(*args, **kwargs):
    return wrapped_transformation_fn(sparsity_func.eval, *args, **kwargs)


def remove_parametrization(*args, **kwargs):
    return wrapped_transformation_fn(sparsity_func.remove_parametrization, *args, **kwargs)


def insert_parametrization(*args, **kwargs):
    return wrapped_transformation_fn(sparsity_func.insert_parametrization, *args, **kwargs)


def get_layer_sparsity_ratio(*args, **kwargs):
    return wrapped_transformation_fn(sparsity_func.get_layer_sparsity_ratio, *args, **kwargs)
