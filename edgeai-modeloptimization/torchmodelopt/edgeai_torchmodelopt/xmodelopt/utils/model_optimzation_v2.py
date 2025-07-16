from .optimization_base import ModelOptimizationBaseModule
from copy import deepcopy


class ModelOptimizationWrapperV2(ModelOptimizationBaseModule):
    def __init__(self, model, *args, example_inputs=None, example_kwargs=None, model_surgery_kwargs=None, pruning_kwargs=None, quantization_kwargs=None, transformation_dict=None, copy_attrs=None, **kwargs):
        copy_attrs = copy_attrs or []
        from ..surgery.v2 import SurgeryModule 
        from ..pruning.v2 import PrunerModule
        from ..quantization.v2.quant_module import QATFxModule, PTCFxModule
        
        if quantization_kwargs:
            quantization_kwargs = deepcopy(quantization_kwargs)
            quantization_method = quantization_kwargs.pop('quantization_method', 'QAT')
            
            if quantization_method == 'QAT':
                quantization_cls = QATFxModule
            elif quantization_method in ('PTC', 'PTQ'):
                quantization_cls = PTCFxModule
        else:
            quantization_cls = QATFxModule
        
        super().__init__(
            model,
            SurgeryModule,
            PrunerModule,
            quantization_cls,
            *args,
            example_inputs=example_inputs,
            example_kwargs=example_kwargs,
            model_surgery_kwargs=model_surgery_kwargs,
            pruning_kwargs=pruning_kwargs,
            quantization_kwargs=quantization_kwargs,
            transformation_dict=transformation_dict,
            copy_attrs=copy_attrs,
            **kwargs
        )