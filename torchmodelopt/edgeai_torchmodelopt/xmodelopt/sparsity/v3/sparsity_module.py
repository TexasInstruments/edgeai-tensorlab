#################################################################################
# Copyright (c) 2018-2023, Texas Instruments Incorporated - http://www.ti.com
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################


import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.utils.source_matcher_utils import  SourcePartition
import torch.nn.utils.parametrize as parametrize
from torch.ao.quantization import quantize_fx
import copy

from . import sparsity_func_wrapper
from ...utils.optimization_base import OptimizationBaseModule
from .parametrization import SPARSITY_CLASS_DICT

class SparserModule(OptimizationBaseModule):
    """Module for applying sparsity to PyTorch models.
    
    This class provides a high-level interface for applying sparsity techniques
    to PyTorch models. It wraps the core sparsity functionality and handles the 
    configuration and lifecycle of the sparsified model.
    
    SparserModule inherits from OptimizationBaseModule, which provides common 
    functionality for model optimization techniques.
    """
    
    def __init__(self, module, *args, example_inputs:list=None, example_kwargs:dict=None, sparsity_ratio=None, total_epochs=None, p=2.0, sparsity_global=False, copy_args=None,
            sparsity_type='n2m', sparsity_init_train_ep=5, sparsity_m=None, add_methods=True, copy_attrs=None, filter_func_register=None, weight_func_register=None, transformation_dict=None, **kwargs) -> None:
        """Initializes a SparserModule.
        
        Args:
            module: The PyTorch module to apply sparsity to.
            *args: Additional arguments passed to the parent class.
            example_inputs (list, optional): Example inputs for model export and tracing. Defaults to None.
            example_kwargs (dict, optional): Example keyword arguments for model export. Defaults to None.
            sparsity_ratio (float, optional): Target sparsity ratio (e.g., 0.5 for 50% sparsity). Defaults to None.
            total_epochs (int, optional): Total number of epochs for sparsity training. Defaults to None.
            p (float, optional): Power parameter for sparsity calculation. Defaults to 2.0.
            sparsity_global (bool, optional): Whether to apply global sparsity across all layers. Defaults to False.
            copy_args (list, optional): List of arguments to copy from the original module. Defaults to None.
            sparsity_type (str, optional): Type of sparsity pattern ('n2m' or 'unstructured'). Defaults to 'n2m'.
            sparsity_init_train_ep (int, optional): Initial number of epochs before sparsification begins. Defaults to 5.
            sparsity_m (int, optional): The m value in n:m sparsity pattern. Defaults to None.
            add_methods (bool, optional): Whether to add sparsity methods to the module. Defaults to True.
            copy_attrs (list, optional): List of attributes to copy from the original module. Defaults to None.
            filter_func_register (function, optional): Custom function to register sparsity filters. Defaults to None.
            weight_func_register (function, optional): Custom function to register weight functions. Defaults to None.
            transformation_dict (dict, optional): Dictionary of transformation functions. Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        copy_attrs = copy_attrs or []
        copy_args = copy_args or []
        example_inputs =[] if example_inputs is None else example_inputs
        example_kwargs = example_kwargs or {}
        super().__init__( module, *args, transformation_dict=transformation_dict, copy_attrs=copy_attrs, **kwargs)
        self.prepare(module, *args, example_inputs=example_inputs, example_kwargs=example_kwargs, sparsity_ratio=sparsity_ratio, total_epochs=total_epochs, p=p, sparsity_global=sparsity_global, copy_args=copy_args,
            sparsity_type=sparsity_type, sparsity_init_train_ep=sparsity_init_train_ep, sparsity_m=sparsity_m, add_methods=add_methods, copy_attrs=copy_attrs, filter_func_register=filter_func_register, weight_func_register=weight_func_register, transformation_dict=transformation_dict, **kwargs)

    def prepare(self, module, *args, example_inputs:list=None, example_kwargs:dict=None, sparsity_ratio=None, total_epochs=None, p=2.0, sparsity_global=False, copy_args=None,
            sparsity_type='n2m', sparsity_init_train_ep=5, sparsity_m=None, add_methods=True, copy_attrs=None, filter_func_register=None, weight_func_register=None,  transformation_dict=None, **kwargs):
        """Prepares the module for sparsity training.
        
        This method configures the sparsity parameters, validates them, and initializes
        the module for sparsity training by calling the sparsity_func_wrapper.init function.
        
        Args:
            module: The PyTorch module to prepare for sparsity.
            *args: Additional arguments to pass to the sparsity initialization function.
            example_inputs (list, optional): Example inputs for model export and tracing. Defaults to None.
            example_kwargs (dict, optional): Example keyword arguments for model export. Defaults to None.
            sparsity_ratio (float, optional): Target sparsity ratio. Defaults to None.
            total_epochs (int, optional): Total number of epochs for sparsity training. Defaults to None.
            p (float, optional): Power parameter for sparsity calculation. Defaults to 2.0.
            sparsity_global (bool, optional): Whether to apply global sparsity. Defaults to False.
            copy_args (list, optional): List of arguments to copy from the original module. Defaults to None.
            sparsity_type (str, optional): Type of sparsity pattern ('n2m' or 'unstructured'). Defaults to 'n2m'.
            sparsity_init_train_ep (int, optional): Initial epochs before sparsification begins. Defaults to 5.
            sparsity_m (int, optional): The m value in n:m sparsity pattern. Defaults to None.
            add_methods (bool, optional): Whether to add sparsity methods to the module. Defaults to True.
            copy_attrs (list, optional): List of attributes to copy from the original module. Defaults to None.
            filter_func_register (function, optional): Custom function to register sparsity filters. Defaults to None.
            weight_func_register (function, optional): Custom function to register weight functions. Defaults to None.
            transformation_dict (dict, optional): Dictionary of transformation functions. Defaults to None.
            **kwargs: Additional keyword arguments for sparsity initialization.
            
        Raises:
            RuntimeError: If required parameters (sparsity_ratio, total_epochs, or sparsity_m for n:m sparsity) 
                are missing or if incompatible options are specified.
        """
        copy_attrs = copy_attrs or []
        copy_args = copy_args or []
        example_inputs =[] if example_inputs is None else example_inputs
        example_kwargs = example_kwargs or {}

        self.epoch_count = 0
        self.sparsity_ratio = sparsity_ratio
        self.total_epochs = total_epochs
        self.sparsity = 0
        self.init_train_ep = sparsity_init_train_ep
        self.p = p
        
        if sparsity_ratio==0:
            raise RuntimeError("sparsity ratio of 0 is not supported , try turning off sparsity and trying again")
        if not(sparsity_ratio and total_epochs):
            raise RuntimeError("sparsity ratio and total epochs are necessary to be provided")
        elif not(sparsity_ratio):
            raise RuntimeError("sparsity ratio should be provided")
        elif not(total_epochs):
            raise RuntimeError("total epochs should be provided")
            
        self.sparsity_class = SPARSITY_CLASS_DICT[sparsity_type]
        
        self.n2m_sparsity = False
        self.unstructured = False
        self.parametrized_params = set()
        
        if sparsity_type=='n2m':
            self.n2m_sparsity = True
        else:
            self.unstructured = True
        self.global_sparsity = sparsity_global
        
        if self.n2m_sparsity:
            if sparsity_m is None:
                raise RuntimeError("The value of m should be provided in case of n:m sparsity")
            else:
                self.m = sparsity_m
        else:
            self.m = None
        
        
        if self.n2m_sparsity and self.global_sparsity:
            print("Cannot do both global sparsity along with n2m sparsity, it doesn't make sense! \n")
            raise NotImplementedError
        
        self.module = sparsity_func_wrapper.init(module, *args, example_inputs=example_inputs, example_kwargs=example_kwargs, sparsity_ratio=sparsity_ratio, total_epochs=total_epochs, p=p, sparsity_global=sparsity_global, copy_args=copy_args,
            sparsity_type=sparsity_type, sparsity_init_train_ep=sparsity_init_train_ep, sparsity_m=sparsity_m, add_methods=add_methods, copy_attrs=copy_attrs, filter_func_register=filter_func_register, weight_func_register=weight_func_register, transformation_dict=transformation_dict,**kwargs)

    #TODO pt2e implementation
    def get_layer_sparsity_ratio(self, sparsity_ratio=0.6):
        """Calculates the sparsity ratio for each layer in the module.
        
        This method calculates appropriate sparsity ratios for individual layers
        within the network, rather than applying a uniform ratio across all layers.
        
        Args:
            sparsity_ratio (float, optional): The target global sparsity ratio. Defaults to 0.6.
            
        Returns:
            SparserModule: Self for method chaining.
            
        Note:
            This is currently a placeholder method for future implementation
            with PyTorch 2.0 (PT2E).
            
        TODO: Implement layer-wise sparsity ratio calculation with PT2E support.
        """
        self.module = sparsity_func_wrapper.get_layer_sparsity_ratio(
            self.module, sparsity_ratio, transformation_dict=self.transformation_dict
        )
        return self
    

    def train(self, mode: bool = True, **kwargs): 
        """Sets the module to training mode with sparsity handling.
        
        This method overrides the standard train() method to handle sparsity parametrization
        during training. It sets the training flag and applies the appropriate sparsity
        function based on the mode.
        
        Args:
            mode (bool, optional): Whether to set the module to training mode. Defaults to True.
            **kwargs: Additional keyword arguments passed to the sparsity functions.
            
        Returns:
            SparserModule: Self for method chaining.
            
        Note:
            The implementation directly sets self.training rather than calling super().train()
            to avoid duplicate calls to the underlying module's train method.
        """
        # this super().train will call all submodules train() wich includes self.module
        # that will effectively call sparsity_func.train with self.module twice
        # to avoid that we directly set self.training
        # super().train(mode)
        self.training = mode

        if mode:
            # Apply train mode with sparsity handling
            self.module = sparsity_func_wrapper.train(
                self.module, mode, transformation_dict=self.transformation_dict, **kwargs
            )
        else:
            # Apply eval mode with sparsity handling
            self.module = sparsity_func_wrapper.eval(
                self.module, mode, transformation_dict=self.transformation_dict, **kwargs
            )
        
        return self
        
    def forward(self, *args, **kwargs):
        """Performs the forward pass through the sparsified module.
        
        This method delegates the forward pass to the underlying sparsified module,
        passing through any arguments and keyword arguments.
        
        Args:
            *args: Arguments to pass to the module's forward method.
            **kwargs: Keyword arguments to pass to the module's forward method.
            
        Returns:
            The output of the module's forward pass.
        """
        return self.module(*args, **kwargs)
        
    def insert_parametrization(self, binary_mask=False):
        """Inserts sparsity parametrization into the module's parameters.
        
        This method adds sparsity parametrization to the module's weight parameters.
        For each identified node/layer, it calculates the appropriate mask and registers
        it over the weights.
        
        Args:
            binary_mask (bool, optional): Whether to use binary masks for sparsity.
                When True, creates hard masks for final sparsification.
                When False, uses soft masks during training. Defaults to False.
                
        Returns:
            SparserModule: Self for method chaining.
        """
        # for each of the nodes/layers, we calculate the parametrization/ mask and then register it over the weights and biases
        self.module = sparsity_func_wrapper.insert_parametrization(
            self.module, binary_mask=binary_mask, transformation_dict=self.transformation_dict
        )
        return self
    
    def remove_parametrization(self, leave_parameterized=True):
        """Removes parametrization from the module's parameters.
        
        This method removes sparsity parametrization from the module's parameters,
        optionally keeping either the original or parametrized values.
        
        Args:
            leave_parameterized (bool, optional): Whether to keep the parametrized values (True)
                or revert to the original values (False). Defaults to True.
                
        Returns:
            SparserModule: Self for method chaining.
            
        Note:
            When leave_parameterized=True (default), the sparsified weights are kept.
            When leave_parameterized=False, the original unsparsified weights are restored.
        """
        # leave_parametrized=False would leave the original parameter instead of the parametrized parameter
        self.module = sparsity_func_wrapper.remove_parametrization(
            self.module, leave_parameterized=leave_parameterized, 
            transformation_dict=self.transformation_dict
        )
        return self
    
    def calculate_sparsity(self):
        """Calculates the current sparsity level of the module.
        
        This method computes the overall sparsity ratio of the module by counting
        the number of zero elements in parametrized tensors and dividing by the
        total number of elements.
        
        Returns:
            SparserModule: Self for method chaining.
            
        Note:
            The calculated sparsity is stored in the module's __sparse_params__.sparsity attribute.
        """
        self.module = sparsity_func_wrapper.calculate_sparsity(
            self.module, transformation_dict=self.transformation_dict
        )
        return self
