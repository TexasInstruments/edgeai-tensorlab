from ...utils.transformation_utils import wrapped_transformation_fn
from ...utils.hooks import add_example_args_kwargs
from . import pruning_func


def init(module, example_inputs, example_kwargs=None, *args, transformation_dict=None, **kwargs):
    example_kwargs = example_kwargs or {}
    add_example_args_kwargs(module, example_inputs=example_inputs, example_kwargs=example_kwargs, transformation_dict=transformation_dict)
    return wrapped_transformation_fn(pruning_func.init,module, example_inputs=example_inputs, example_kwargs=example_kwargs, transformation_dict=transformation_dict, **kwargs)


def calculate_sparsity(*args, **kwargs):
    return wrapped_transformation_fn(pruning_func.calculate_sparsity, *args, **kwargs)


def train(*args, **kwargs):
    return wrapped_transformation_fn(pruning_func.train, *args, **kwargs)


def remove_parametrization(*args, **kwargs):
    return wrapped_transformation_fn(pruning_func.remove_parametrization, *args, **kwargs)


def insert_parametrization(*args, **kwargs):
    return wrapped_transformation_fn(pruning_func.insert_parametrization, *args, **kwargs)


def get_layer_pruning_ratio(*args, **kwargs):
    return wrapped_transformation_fn(pruning_func.get_layer_pruning_ratio, *args, **kwargs)


def get_layer_pruning_ratio_channel(*args, **kwargs):
    return wrapped_transformation_fn(pruning_func.get_layer_pruning_ratio_channel, *args, **kwargs)