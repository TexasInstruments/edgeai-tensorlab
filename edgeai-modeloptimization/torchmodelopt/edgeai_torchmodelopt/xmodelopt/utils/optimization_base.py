from torch import nn
import torch
import types
import warnings
from copy import deepcopy
from .helper_functions import _model_to_device
from .transformation_utils import wrapped_transformation_fn

def add_attrs(self, attrs, src):
    """Adds attributes from a source object or class to a target object.
    
    Args:
        self: The target object to add attributes to.
        attrs (list): List of attribute names to copy from the source.
        src: The source object or class to copy attributes from.
        
    Warnings:
        If src is a class, non-method attributes (instance attributes) can't be copied
        properly and will be discarded.
    """
    # Determine if source is a class or object
    if isinstance(src, type):
        warnings.warn(f"{src.__name__} is a class. Some of the attributes (non-methods) in 'attrs' may not be of a class. They may be attributes of an object. If they are passed, they will be discarded.")
        src_cls = src
    else:
        src_cls = src.__class__

    def create_function(fn_name):
        """Creates a method that delegates to a source class method."""
        def func(self, *args, **kwargs):
            fn = getattr(src_cls, fn_name)
            return fn(self, *args, **kwargs)
        f = types.MethodType(func, self)
        setattr(self, fn_name, f)

    def create_property(name):
        """Creates a property that delegates to a source object attribute."""
        attribute_getter = lambda self: getattr(src, name)
        attribute_setter = lambda self, value: setattr(src, name, value)
        new_property = property(fget=attribute_getter, fset=attribute_setter)
        setattr(self.__class__, name, new_property)

    # Process each attribute in the list
    for attr_name in attrs:
        attr = getattr(src, attr_name, None)
        if attr:
            if isinstance(attr, (types.MethodType, types.FunctionType)):
                create_function(attr_name)
            else:
                create_property(attr_name)


class OptimizationBaseModule(nn.Module):
    """Base module for optimization operations on PyTorch models.
    
    This class serves as a foundation for different types of model optimizations,
    providing common functionality and a standardized interface for optimization
    modules.
    
    Note:
        Derived classes should implement the prepare method with appropriate
        optimization logic.
    """
    
    def __init__(self, model, *args, transformation_dict=None, copy_attrs=None, **kwargs):
        """Initializes the OptimizationBaseModule.
        
        Args:
            model (nn.Module): The model to optimize.
            *args: Additional positional arguments.
            transformation_dict (dict, optional): Dictionary mapping module names to
                transformation functions. Defaults to None.
            copy_attrs (list, optional): List of attribute names to copy from the
                original model. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        copy_attrs = copy_attrs or []
        super().__init__()
        # Store original and working copies of the model
        self._orig_module = deepcopy(model)
        self.module = model
        self.transformation_dict = transformation_dict
        add_attrs(self, copy_attrs, self.module)

    def prepare(self, *args, **kwargs):
        """Prepares the model for optimization.
        
        This method should be implemented by derived classes to apply specific
        optimization techniques to the model.
        
        Args:
            *args: Positional arguments for the preparation process.
            **kwargs: Keyword arguments for the preparation process.
            
        Raises:
            NotImplementedError: This method must be implemented by derived classes.
        """
        raise NotImplementedError('prepare method needs to be implemented')

    @classmethod
    def _add_attrs_to(cls, obj, attr_names=None):
        """Adds class attributes to an optimization module instance.
        
        Args:
            obj (OptimizationBaseModule): The object to add attributes to.
            attr_names (list, optional): List of attribute names to add.
                Defaults to None (empty list).
                
        Raises:
            AssertionError: If obj is not an instance of OptimizationBaseModule.
        """
        attr_names = attr_names or []
        assert isinstance(obj, OptimizationBaseModule), 'This only works if self is an OptimizationBaseModule object'
        add_attrs(obj, attr_names, cls)

    def forward(self, *args, **kwargs):
        """Forwards the call to the wrapped module.
        
        Args:
            *args: Positional arguments to pass to the wrapped module.
            **kwargs: Keyword arguments to pass to the wrapped module.
            
        Returns:
            The output of the wrapped module's forward method.
        """
        return self.module(*args, **kwargs)
    
    def to(self, *args, **kwargs):
        """Moves the module and wrapped model to the specified device or dtype.
        
        This method extends the standard nn.Module.to() method by also moving the
        wrapped module to the same device using the _model_to_device function
        with appropriate transformations.
        
        Args:
            *args: Positional arguments specifying device or dtype.
            **kwargs: Keyword arguments which may include 'device' parameter.
            
        Returns:
            Self with both the wrapper and wrapped module moved to the specified device.
        """
        # First move the wrapper module (self) to the device/dtype
        self = super().to(*args, **kwargs)
        
        # Extract the target device from args or kwargs
        device = kwargs.get('device', None) or args[0]
        
        # Move the wrapped module to the device using transformation-aware function
        self.module = wrapped_transformation_fn(_model_to_device, self.module, 
                                               transformation_dict=self.transformation_dict, 
                                               device=device)
        return self
    
    def cpu(self):
        """Moves the module and wrapped model to CPU.
        
        This method extends the standard nn.Module.cpu() method by also moving the
        wrapped module to CPU using the _model_to_device function with appropriate
        transformations.
        
        Returns:
            Self with both the wrapper and wrapped module moved to CPU.
        """
        # First move the wrapper module (self) to CPU
        self = super().cpu()
        
        # Move the wrapped module to CPU using transformation-aware function
        self.module = wrapped_transformation_fn(_model_to_device, self.module, 
                                               transformation_dict=self.transformation_dict, 
                                               device='cpu')
        return self
    
    def cuda(self, device=None):
        """Moves the module and wrapped model to CUDA device.
        
        This method extends the standard nn.Module.cuda() method by also moving the
        wrapped module to the specified CUDA device using the _model_to_device
        function with appropriate transformations.
        
        Args:
            device: CUDA device index or string. If None, uses the current CUDA device.
            
        Returns:
            Self with both the wrapper and wrapped module moved to the specified CUDA device.
        """
        # First move the wrapper module (self) to CUDA
        self = super().cuda(device)
        
        # Use specified device or default to 'cuda' (current device)
        device = device or 'cuda'
        
        # Move the wrapped module to the CUDA device using transformation-aware function
        self.module = wrapped_transformation_fn(_model_to_device, self.module, 
                                               transformation_dict=self.transformation_dict, 
                                               device=device)
        return self


class ModelOptimizationBaseModule(OptimizationBaseModule):
    """Base module that combines multiple optimization techniques for a model.
    
    This class extends OptimizationBaseModule to support multiple optimization
    techniques: model surgery, pruning, and quantization.
    """
    
    def __init__(self, model, surgery_module_cls, pruning_module_cls, quantization_module_cls, *args, 
                 example_inputs=None, example_kwargs=None, model_surgery_kwargs=None, pruning_kwargs=None, 
                 quantization_kwargs=None, transformation_dict=None, copy_attrs=None, **kwargs):
        """Initializes the ModelOptimizationBaseModule.
        
        Args:
            model (nn.Module): The model to optimize.
            surgery_module_cls: Class implementing model surgery functionality.
            pruning_module_cls: Class implementing pruning functionality.
            quantization_module_cls: Class implementing quantization functionality.
            *args: Additional positional arguments.
            example_inputs (list, optional): Example inputs for the model.
            example_kwargs (dict, optional): Example keyword arguments for the model.
            model_surgery_kwargs (dict, optional): Keyword arguments for model surgery.
            pruning_kwargs (dict, optional): Keyword arguments for pruning.
            quantization_kwargs (dict, optional): Keyword arguments for quantization.
            transformation_dict (dict, optional): Dictionary mapping module names to
                transformation functions.
            copy_attrs (list, optional): List of attribute names to copy.
            **kwargs: Additional keyword arguments.
        """
        copy_attrs = copy_attrs or []
        super().__init__(model, *args, transformation_dict=transformation_dict, copy_attrs=copy_attrs, **kwargs)
        # Store optimization parameters
        self.model_surgery_kwargs = model_surgery_kwargs or {}
        self.pruning_kwargs = pruning_kwargs or {}
        self.quantization_kwargs = quantization_kwargs or {}
        self.example_inputs = example_inputs or []
        self.example_kwargs = example_kwargs or {}
        # Store optimization module classes
        self.surgery_module_cls = surgery_module_cls
        self.pruning_module_cls = pruning_module_cls
        self.quantization_module_cls = quantization_module_cls
        # Prepare the model with the specified optimization techniques
        self.prepare(self.module, *args, example_inputs=self.example_inputs, example_kwargs=self.example_kwargs, 
                     model_surgery_kwargs=self.model_surgery_kwargs, pruning_kwargs=self.pruning_kwargs, 
                     quantization_kwargs=self.quantization_kwargs, transformation_dict=self.transformation_dict, copy_attrs=copy_attrs, **kwargs)

    def prepare(self, model, *args, example_inputs=None, example_kwargs=None, model_surgery_kwargs=None, pruning_kwargs=None, 
                quantization_kwargs=None, transformation_dict=None, copy_attrs=None, **kwargs):
        """Prepares the model by applying model surgery, pruning, and quantization.
        
        This method coordinates the application of different optimization techniques
        to the model based on the provided configuration.
        
        Args:
            model (nn.Module): The model to prepare for optimization.
            *args: Additional positional arguments.
            example_inputs (list, optional): Example inputs for the model.
            example_kwargs (dict, optional): Example keyword arguments for the model.
            model_surgery_kwargs (dict, optional): Keyword arguments for model surgery.
            pruning_kwargs (dict, optional): Keyword arguments for pruning.
            quantization_kwargs (dict, optional): Keyword arguments for quantization.
            transformation_dict (dict, optional): Dictionary mapping module names to
                transformation functions.
            copy_attrs (list, optional): List of attribute names to copy.
            **kwargs: Additional keyword arguments.
            
        Raises:
            AssertionError: If self is not an instance of OptimizationBaseModule.
        """
        assert isinstance(self, OptimizationBaseModule), 'This only works for OptimizationBaseModule objects'
        copy_attrs = copy_attrs or []
        self.module = model

        # Apply model surgery if requested
        if model_surgery_kwargs:
            model_surgery_kwargs = deepcopy(model_surgery_kwargs)
            self.surgery_module_cls.prepare(self, self.module, example_inputs=example_inputs, example_kwargs=example_kwargs, 
                                            transformation_dict=transformation_dict, copy_attrs=copy_attrs, **model_surgery_kwargs, **kwargs)
            self.surgery_module_cls._add_attrs_to(self)

        # Apply pruning if requested
        if pruning_kwargs:
            pruning_kwargs = deepcopy(pruning_kwargs)
            self.pruning_module_cls.prepare(self, self.module, example_inputs=example_inputs, example_kwargs=example_kwargs, 
                                            transformation_dict=transformation_dict, copy_attrs=copy_attrs, **pruning_kwargs, **kwargs)

        # Apply quantization if requested
        if quantization_kwargs:
            quantization_kwargs = deepcopy(quantization_kwargs)
            self.quantization_module_cls.prepare(self, self.module, example_inputs=example_inputs, example_kwargs=example_kwargs, 
                                                 transformation_dict=transformation_dict, copy_attrs=copy_attrs, **quantization_kwargs, **kwargs)
            self.quantization_module_cls._add_attrs_to(self)

    def train(self, *args, **kwargs):
        """Sets the module in training mode.
        
        This method overrides the standard train method to handle training mode
        appropriately for optimized models, delegating to the appropriate optimization
        module's train method if available.
        
        Args:
            *args: Positional arguments for the training mode.
            **kwargs: Keyword arguments for the training mode.
            
        Returns:
            The module in training mode.
        """
        # Delegate to appropriate optimization module's train method
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
        """Adds class attributes to an optimization module instance.
        
        This class method overrides the base class method to add default attributes
        for conversion and export functionality.
        
        Args:
            obj (OptimizationBaseModule): The object to add attributes to.
            attr_names (list, optional): List of attribute names to add.
                Defaults to ['convert', 'export'].
        """
        attr_names = attr_names or ['convert', 'export']
        super()._add_attrs_to(obj, attr_names)

    def convert(self, *args, **kwargs):
        """Converts the model according to the applied optimizations.
        
        This method delegates to the quantization module's convert method if
        quantization was applied, otherwise returns the model unchanged.
        
        Args:
            *args: Positional arguments for the conversion.
            **kwargs: Keyword arguments for the conversion.
            
        Returns:
            The converted model.
        """
        if self.quantization_kwargs:
            return self.quantization_module_cls.convert(self, *args, **kwargs)
        return self

    def export(self, *args, **kwargs):
        """Exports the model to ONNX format.
        
        This method delegates to the quantization module's export method if
        quantization was applied, otherwise uses the standard PyTorch ONNX export.
        
        Args:
            *args: Positional arguments for the export.
            **kwargs: Keyword arguments for the export.
        """
        if self.quantization_kwargs:
            self.quantization_module_cls.export(self, *args, **kwargs)
        else:
            torch.onnx.export(self, *args, **kwargs)