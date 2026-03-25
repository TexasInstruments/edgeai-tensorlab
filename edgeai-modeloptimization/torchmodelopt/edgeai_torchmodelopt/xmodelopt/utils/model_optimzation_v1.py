from .optimization_base import ModelOptimizationBaseModule


class ModelOptimizationWrapperV1(ModelOptimizationBaseModule):
    """Version 1 of the model optimization wrapper.
    
    This class extends ModelOptimizationBaseModule to implement version 1 of the
    model optimization techniques. It uses the version 1 implementation of surgery
    and quantization modules.
    
    Note:
        Version 1 does not support pruning, so None is passed as the pruning module.
    """
    
    def __init__(self, model, *args, example_inputs=None, example_kwargs=None, model_surgery_kwargs=None, pruning_kwargs=None, quantization_kwargs=None, transformation_dict=None, copy_attrs=None, **kwargs):
        """Initializes the ModelOptimizationWrapperV1.
        
        Args:
            model (nn.Module): The model to optimize.
            *args: Additional positional arguments.
            example_inputs (list, optional): Example inputs for the model. Defaults to None.
            example_kwargs (dict, optional): Example keyword arguments for the model. Defaults to None.
            model_surgery_kwargs (dict, optional): Keyword arguments for model surgery. Defaults to None.
            pruning_kwargs (dict, optional): Keyword arguments for pruning. Defaults to None.
            quantization_kwargs (dict, optional): Keyword arguments for quantization. Defaults to None.
            transformation_dict (dict, optional): Dictionary mapping module names to transformation functions. Defaults to None.
            copy_attrs (list, optional): List of attribute names to copy. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        copy_attrs = copy_attrs or []        
        from ..experimental.surgery.v1 import SurgeryModule 
        from ..quantization.v1 import QuantTrainModule
        
        super().__init__(
            model,
            SurgeryModule,
            None,  # Version 1 does not support pruning
            QuantTrainModule,  # Fixed syntax error: added comma
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