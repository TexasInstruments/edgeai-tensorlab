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
        if isinstance(module, nn.ModuleList):  # not well tested, example inputs might not be passed in correct manner
            for i, mod in enumerate(module):
                if isinstance(mod, nn.ModuleList): # taken care of 2 nested modulelist, need to generalise over many #TODO
                    for j, mo in enumerate(mod):
                        if mo is not None:
                            mo = wrapper_fn(mo, *args, **kwargs)
                        mod[j] = mo
                    #
                #
                else:
                    if mod is not None:
                        mod = wrapper_fn(mod, *args, **kwargs)
                module[i] = mod
                # example_inputs = kwargs.get("example_inputs", None)
                # if example_inputs is not None: #or kwargs.get("example_kwargs", None): TODO deal with kwargs passed to modulelist
                #     example_inputs = [mod(*example_inputs)]
                # mod = wrapper_fn(mod, *args, **kwargs)
                # if example_inputs is not None:
                #     kwargs["example_inputs"] = example_inputs
            #
        #
        else:
            module = wrapper_fn(module, *args, **kwargs)
        setattr(parent_module, sub_module_name, module)
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
        return apply_tranformation_to_submodules(model, transformation_dict, *args, **kwargs)
    else:
        return fn(model, *args, **kwargs)
    