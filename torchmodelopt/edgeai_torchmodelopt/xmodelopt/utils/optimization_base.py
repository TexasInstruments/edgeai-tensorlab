from torch import nn
import torch
import types
import warnings
from copy import deepcopy


def add_attrs(self, attrs, src):
    if isinstance(src, type):
        warnings.warn(f"{src.__name__} is a class. Some of the attributes (non-methods) in 'attrs' may not be of a class. They may be attributes of an object. If they are passed, they will be discarded.")
        src_cls = src
    else:
        src_cls = src.__class__

    def create_function(fn_name):
        def func(self, *args, **kwargs):
            fn = getattr(src_cls, fn_name)
            return fn(self, *args, **kwargs)
        f = types.MethodType(func, self)
        setattr(self, fn_name, f)

    def create_property(name):
        attribute_getter = lambda self: getattr(src, name)
        attribute_setter = lambda self, value: setattr(src, name, value)
        new_property = property(fget=attribute_getter, fset=attribute_setter)
        setattr(self.__class__, name, new_property)

    for attr_name in attrs:
        attr = getattr(src, attr_name, None)
        if attr:
            if isinstance(attr, (types.MethodType, types.FunctionType)):
                create_function(attr_name)
            else:
                create_property(attr_name)


class OptimizationBaseModule(nn.Module):
    def __init__(self, model, *args, transformation_dict=None, copy_attrs=None, **kwargs):
        copy_attrs = copy_attrs or []
        super().__init__()
        self._orig_module = deepcopy(model)
        self.module = model
        self.transformation_dict = transformation_dict
        add_attrs(self, copy_attrs, self.module)

    def prepare(self, *args, **kwargs):
        raise NotImplementedError('prepare method needs to be implemented')

    @classmethod
    def _add_attrs_to(cls, obj, attr_names=None):
        attr_names = attr_names or []
        assert isinstance(obj, OptimizationBaseModule), 'This only works if self is an OptimizationBaseModule object'
        add_attrs(obj, attr_names, cls)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ModelOptimizationBaseModule(OptimizationBaseModule):
    def __init__(self, model, surgery_module_cls, pruning_module_cls, quantization_module_cls, *args, 
                 example_inputs=None, example_kwargs=None, model_surgery_kwargs=None, pruning_kwargs=None, 
                 quantization_kwargs=None, transformation_dict=None, copy_attrs=None, **kwargs):
        copy_attrs = copy_attrs or []
        super().__init__(model, *args, transformation_dict=transformation_dict, copy_attrs=copy_attrs, **kwargs)
        self.model_surgery_kwargs = model_surgery_kwargs or {}
        self.pruning_kwargs = pruning_kwargs or {}
        self.quantization_kwargs = quantization_kwargs or {}
        self.example_inputs = example_inputs or []
        self.example_kwargs = example_kwargs or {}
        self.surgery_module_cls = surgery_module_cls
        self.pruning_module_cls = pruning_module_cls
        self.quantization_module_cls = quantization_module_cls
        self.prepare(self.module, *args, example_inputs=self.example_inputs, example_kwargs=self.example_kwargs, 
                     model_surgery_kwargs=self.model_surgery_kwargs, pruning_kwargs=self.pruning_kwargs, 
                     quantization_kwargs=self.quantization_kwargs, transformation_dict=self.transformation_dict, copy_attrs=copy_attrs, **kwargs)

    def prepare(self, model, *args, example_inputs=None, example_kwargs=None, model_surgery_kwargs=None, pruning_kwargs=None, 
                quantization_kwargs=None, transformation_dict=None, copy_attrs=None, **kwargs):
        assert isinstance(self, OptimizationBaseModule), 'This only works for OptimizationBaseModule objects'
        copy_attrs = copy_attrs or []
        self.module = model

        if model_surgery_kwargs:
            model_surgery_kwargs = deepcopy(model_surgery_kwargs)
            self.surgery_module_cls.prepare(self, self.module, example_inputs=example_inputs, example_kwargs=example_kwargs, 
                                            transformation_dict=transformation_dict, copy_attrs=copy_attrs, **model_surgery_kwargs, **kwargs)
            self.surgery_module_cls._add_attrs_to(self)

        if pruning_kwargs:
            pruning_kwargs = deepcopy(pruning_kwargs)
            self.pruning_module_cls.prepare(self, self.module, example_inputs=example_inputs, example_kwargs=example_kwargs, 
                                            transformation_dict=transformation_dict, copy_attrs=copy_attrs, **pruning_kwargs, **kwargs)

        if quantization_kwargs:
            quantization_kwargs = deepcopy(quantization_kwargs)
            self.quantization_module_cls.prepare(self, self.module, example_inputs=example_inputs, example_kwargs=example_kwargs, 
                                                 transformation_dict=transformation_dict, copy_attrs=copy_attrs, **quantization_kwargs, **kwargs)
            self.quantization_module_cls._add_attrs_to(self)

    def train(self, *args, **kwargs):
        if self.pruning_kwargs or self.quantization_kwargs:
            if self.quantization_kwargs:
                self = self.quantization_module_cls.train(self, *args, **kwargs)
            elif self.pruning_kwargs:
                self = self.pruning_module_cls.train(self, *args, **kwargs)
        else:
            self = super().train(*args, **kwargs)
        return self

    @classmethod
    def _add_attrs_to(cls, obj, attr_names=None):
        attr_names = attr_names or ['convert', 'export']
        super()._add_attrs_to(obj, attr_names)

    def convert(self, *args, **kwargs):
        if self.quantization_kwargs:
            return self.quantization_module_cls.convert(self, *args, **kwargs)
        return self

    def export(self, *args, **kwargs):
        if self.quantization_kwargs:
            self.quantization_module_cls.export(self, *args, **kwargs)
        else:
            torch.onnx.export(self, *args, **kwargs)