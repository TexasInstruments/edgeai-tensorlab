from .optimization_base import ModelOptimizationBaseModule


class ModelOptimizationWrapperV1(ModelOptimizationBaseModule):
    def __init__(self, model, *args, example_inputs=None, example_kwargs=None, model_surgery_kwargs=None, pruning_kwargs=None, quantization_kwargs=None, transformation_dict=None, copy_attrs=None, **kwargs):
        copy_attrs = copy_attrs or []        
        from ..surgery.v1 import SurgeryModule 
        from ..quantization.v1 import QuantTrainModule
        
        super().__init__(
            model,
            SurgeryModule,
            None,
            QuantTrainModule
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