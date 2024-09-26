from torch import nn
import torch
import types
import warnings
from copy import deepcopy


def add_attrs(self, attrs , src):
    if isinstance(src, type):
        warnings.warn(f'''
            {src.__name__} is a class. Some of the attributes(non-methods) in 'attrs' may not be of a class. 
            They may be attributes of an object. If they are passed, they will be discarded.
            ''')
        src_cls = src
    else:
        src_cls = src.__class__
    
    def create_function(fn_name):
        def func(self, *args, **kwargs):
            fn = getattr(src_cls, fn_name)
            return fn(self, *args, **kwargs)
        f=  types.MethodType(func, self)
        setattr(self, fn_name, f)
    
    def create_property(name):
        attribute_getter = lambda self: getattr(src, name)
        attribute_setter = lambda self, value: setattr(src, name, value)
        new_property = property(fget=attribute_getter, fset=attribute_setter)
        new_property=  types.MethodType(lambda *args, **kwargs :new_property, self)
        setattr(self, name, new_property)
    
    for attr_name in attrs:
            attr = getattr(src,attr_name,None)
            if attr:
                if isinstance(attr, (types.MethodType, types.FunctionType)):
                    create_function(attr_name)
                else:
                    create_property(attr_name)


def wrap__load_from_state_dict_method(module:nn.Module):
    method_name = '_load_from_state_dict'
    def wrapped(self, state_dict:dict[str,any], prefix,*args, **kwargs):
        method = getattr(self, method_name)
        if all(key.startswith('module') for key in state_dict):
            return method(state_dict, prefix, *args, **kwargs)
        return method(state_dict, prefix+'module.', *args, **kwargs)
    method = types.MethodType(wrapped, module)
    setattr(module, method_name, method)
    for name, child in module.named_children():
        wrap__load_from_state_dict_method(child)
    return


class OptimizationBaseModule(nn.Module):
    def __init__(self, model,*args, transformation_dict=None, copy_attrs=None, **kwargs) -> None:
        copy_attrs= copy_attrs or []
        super().__init__()
        self.module = model
        self.transformation_dict = transformation_dict
        add_attrs(self, copy_attrs, self.module)
        wrap__load_from_state_dict_method(self)
                    
    def prepare(self, *args, **kwargs):
        raise NotImplementedError('prepare method is needed to be implemented')
    
    def forward(self, *args , **kwargs):
        return self.module(*args, **kwargs)

    def load_state_dict(self, state_dict: types.Mapping[str, torch.Any], strict: bool = True, assign: bool = False):
        if all(key.startswith('module') for key in state_dict):
            return super().load_state_dict(state_dict, prefix, *args, **kwargs)
        return super().load_state_dict(state_dict, prefix+'module.', *args, **kwargs)


class ModelOptimizationBaseModule(OptimizationBaseModule):
    def __init__(self, model, surgery_module_cls, pruning_module_cls, quantization_module_cls, *args, example_inputs=None, example_kwargs=None, model_surgery_kwargs=None, pruning_kwargs=None, quantization_kwargs=None, transformation_dict=None, copy_attrs=None, **kwargs) -> None:
        copy_attrs= copy_attrs or []
        OptimizationBaseModule.__init__(self,model, *args, transformation_dict=transformation_dict, copy_attrs=copy_attrs, *kwargs)
        self.model_surgery_kwargs = model_surgery_kwargs or {}
        self.pruning_kwargs=pruning_kwargs or {}
        self.quantization_kwargs= quantization_kwargs or {}
        self.example_inputs=example_inputs or []
        self.example_kwargs=example_kwargs or {}
        self.surgery_module_cls= surgery_module_cls
        self.pruning_module_cls = pruning_module_cls
        self.quantization__module_cls = quantization_module_cls
        self.prepare(self.module, *args, model_surgery_kwargs=self.model_surgery_kwargs, pruning_kwargs=self.pruning_kwargs, quantization_kwargs=self.quantization_kwargs, transformation_dict=self.transformation_dict, copy_attrs=copy_attrs, **kwargs )
        
    def prepare(self, model, *args, model_surgery_kwargs=None, pruning_kwargs=None, quantization_kwargs=None, transformation_dict=None, copy_attrs=None, **kwargs):
        assert isinstance(self, OptimizationBaseModule), 'This only works for OptimizationBaseModule Objects'
        copy_attrs = copy_attrs or []
        self.module = model
        if model_surgery_kwargs:
            model_surgery_kwargs = deepcopy(model_surgery_kwargs)
            self.surgery_module_cls.prepare(self, self.module, *args, transformation_dict=transformation_dict, copy_attrs=copy_attrs, **model_surgery_kwargs, **kwargs)
            self.surgery_module_cls._add_attrs_to(self)
        if pruning_kwargs:
            pruning_kwargs = deepcopy(pruning_kwargs)
            self.pruning_module_cls.__init__(self, self.module, transformation_dict=transformation_dict, copy_attrs=copy_attrs, **pruning_kwargs, **kwargs)
        if quantization_kwargs:
            quantization_kwargs = deepcopy(quantization_kwargs)
            self.quantization__module_cls.prepare(self, self.module, transformation_dict=transformation_dict, copy_attrs=copy_attrs, **quantization_kwargs, **kwargs)
            self.quantization__module_cls._add_attrs_to(self)

    def train(self, *args, **kwargs):
        if any(opt_kwargs!={} for opt_kwargs in (self.pruning_kwargs,self.quantization_kwargs)):
            if self.pruning_kwargs:
                self = self.pruning_module_cls.train(self, *args, **kwargs)
            if self.quantization_kwargs:
                self = self.quantization__module_cls.train(self, *args, *kwargs)
        else:
            self = OptimizationBaseModule.train(self, *args, **kwargs)
        return self
    
    def convert(self, *args, **kwargs):
        if self.quantization_kwargs:
            self = self.quantization__module_cls.convert(self, *args, **kwargs)
        return self
    
    def export(self, *args, **kwargs):
        if self.quantization_kwargs:
            self.quantization__module_cls.export(self, *args, **kwargs)
        else:
            torch.onnx.export(self, *args, **kwargs)