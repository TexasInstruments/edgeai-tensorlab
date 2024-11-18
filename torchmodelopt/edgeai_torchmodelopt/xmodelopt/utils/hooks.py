import torch
from torch import nn
from .transformation_utils import wrapped_transformation_fn

def detach_all_tensors(vals):
    if isinstance(vals, (list, tuple)):
        return [detach_all_tensors(v) for v in vals]
    elif isinstance(vals, dict):
        return {k: detach_all_tensors(v) for k, v in vals.items()}
    elif isinstance(vals, torch.Tensor):
        return vals.detach()
    else:
        return vals


def record_inputs_pre_hook(self: nn.Module, args, kwargs):
    self._example_inputs = detach_all_tensors(args)
    self._example_kwargs = detach_all_tensors(kwargs)


def register_pre_hook_for_optimization(self: nn.Module):
    if not hasattr(self, '_example_inputs'):
        self.__optimization_pre_hook = self.register_forward_pre_hook(record_inputs_pre_hook, prepend=True, with_kwargs=True)
    return self


def disable_pre_hook_for_optimization(self: nn.Module):
    if hasattr(self, '__optimization_pre_hook'):
        self.__optimization_pre_hook.remove()
        del self.__optimization_pre_hook
    return self


def add_example_args_kwargs(module, example_inputs, example_kwargs=None, transformation_dict=None):
    if (example_inputs is None or example_inputs in ([], [None], (), (None,))) and (example_kwargs is None or example_kwargs == {}):
        return

    example_kwargs = example_kwargs or {}
    if not isinstance(example_inputs, (tuple, list)):
        example_inputs = (example_inputs,)
    wrapped_transformation_fn(register_pre_hook_for_optimization, module, transformation_dict=transformation_dict)
    module.eval()
    module(*example_inputs, **example_kwargs)
    wrapped_transformation_fn(disable_pre_hook_for_optimization, module, transformation_dict=transformation_dict)
    