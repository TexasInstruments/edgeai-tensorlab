import torch

from .optimization_base import ModelOptimizationBaseModule
from ..surgery.v3 import SurgeryModule 
from ..pruning.v3 import PrunerModule 
from ..quantization.v3.quant_module import QuantPT2EBaseModule, QATPT2EModule, PTQPT2EModule
from copy import deepcopy


class ModelOptimizationWrapperV3(ModelOptimizationBaseModule):
    def __init__(self, model, *args, example_inputs=None, example_kwargs=None, model_surgery_kwargs=None, pruning_kwargs=None, quantization_kwargs=None, transformation_dict=None, copy_attrs=None, **kwargs) -> None:
        copy_attrs= copy_attrs or []
        quantization_kwargs = deepcopy(quantization_kwargs)
        quantization_method = quantization_kwargs.pop('quantization_method','QAT')
        if quantization_method == 'QAT':
            quantization_cls = QATPT2EModule
        if quantization_method in ('PTC', 'PTQ'):
            quantization_cls = PTQPT2EModule
        super().__init__(model,SurgeryModule, PrunerModule, quantization_cls, *args, example_inputs=example_inputs, example_kwargs=example_kwargs, model_surgery_kwargs=model_surgery_kwargs, pruning_kwargs=pruning_kwargs, quantization_kwargs=quantization_kwargs, transformation_dict=transformation_dict, copy_attrs=copy_attrs, **kwargs)
        