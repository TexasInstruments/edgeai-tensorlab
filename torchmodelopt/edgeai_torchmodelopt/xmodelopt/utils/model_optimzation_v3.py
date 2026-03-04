from .optimization_base import ModelOptimizationBaseModule
from copy import deepcopy


class ModelOptimizationWrapperV3(ModelOptimizationBaseModule):
    """Version 3 of the model optimization wrapper.
    
    This class extends ModelOptimizationBaseModule to implement version 3 of the
    model optimization techniques. It uses the version 3 implementation of surgery,
    pruning, and quantization modules which leverage PyTorch 2.0 features.
    
    Version 3 is designed to work with PyTorch 2.0's export functionality (PT2E)
    and offers improved performance and compatibility with modern PyTorch features.
    """
    
    def __init__(self, model, *args, example_inputs=None, example_kwargs=None, model_surgery_kwargs=None, pruning_kwargs=None,
                 quantization_kwargs=None, transformation_dict=None, copy_attrs=None, **kwargs):
        """Initializes the ModelOptimizationWrapperV3.
        
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
        from ..experimental.surgery.v3 import SurgeryModule 
        from ..experimental.pruning.v3 import PrunerModule 
        from ..quantization.v3.quant_module import QATPT2EModule, PTQPT2EModule
        
        # Determine which quantization class to use based on the method specified
        if quantization_kwargs:
            quantization_kwargs = deepcopy(quantization_kwargs)
            quantization_method = quantization_kwargs.pop('quantization_method', 'QAT')
            
            if quantization_method == 'QAT':
                quantization_cls = QATPT2EModule
            elif quantization_method in ('PTC', 'PTQ'):
                quantization_cls = PTQPT2EModule
        else:
            # Default to QAT if no quantization kwargs are provided
            quantization_cls = QATPT2EModule
        
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