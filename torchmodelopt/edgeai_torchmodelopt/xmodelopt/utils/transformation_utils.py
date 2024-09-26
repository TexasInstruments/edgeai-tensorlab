from torch import nn
import copy


def apply_tranformation_to_submodules(model:nn.Module, transformation_dict: dict, *args, **kwargs):
    module_dict = dict(model.named_modules())
    for name, wrapper_fn in transformation_dict.items() :
        if name not in module_dict:
            continue
        module = module_dict[name]
        splits = name.rsplit('.',1)
        if len(splits) == 1:
            splits = '',splits[0]
        parent_module, sub_module_name = splits
        parent_module = model if parent_module == '' else module_dict[parent_module]
        module = wrapper_fn(module, *args, **kwargs)
        setattr(parent_module,sub_module_name,module)
    return model


class TransformationWrapper:
    def __init__(self, wrapper:callable, fn:callable=None):
        self.wrapper = wrapper
        self.fn = fn
        
    def __call__(self, *args, **kwargs):
        if self.fn is None:
            raise ValueError("the fn function to be wrapped should not be None")
        return self.wrapper(self.fn, *args, **kwargs)


def wrapped_transformation_fn(fn, model, *args, transformation_dict=None, **kwargs):
    if transformation_dict is not None:
        transformation_dict = copy.deepcopy(transformation_dict)
        for key, value in transformation_dict.items():
            if value is None:
                transformation_dict[key] = fn
            elif isinstance(value, TransformationWrapper) and value.fn is None:
                value.fn = fn
        return apply_tranformation_to_submodules(model,transformation_dict, *args, **kwargs)
    else:
        return fn(model, *args, **kwargs)
    