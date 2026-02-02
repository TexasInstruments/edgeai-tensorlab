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
    def __init__(self, module, *args, example_inputs:list=None, example_kwargs:dict=None, sparsity_ratio=None, total_epochs=None, p=2.0, sparsity_global=False, copy_args=None,
            sparsity_type='n2m', sparsity_init_train_ep=5, sparsity_m=None, add_methods=True, aten_graph=True, copy_attrs=None, filter_func_register=None, weight_func_register=None, transformation_dict=None, **kwargs) -> None:
        copy_attrs = copy_attrs or []
        copy_args = copy_args or []
        example_inputs =[] if example_inputs is None else example_inputs
        example_kwargs = example_kwargs or {}
        super().__init__( module, *args, transformation_dict=transformation_dict, copy_attrs=copy_attrs, **kwargs)
        self.prepare(module, *args, example_inputs=example_inputs, example_kwargs=example_kwargs, sparsity_ratio=sparsity_ratio, total_epochs=total_epochs, p=p, sparsity_global=sparsity_global, copy_args=copy_args,
            sparsity_type=sparsity_type, sparsity_init_train_ep=sparsity_init_train_ep, sparsity_m=sparsity_m, add_methods=add_methods, aten_graph=aten_graph, copy_attrs=copy_attrs, filter_func_register=filter_func_register, weight_func_register=weight_func_register, transformation_dict=transformation_dict, **kwargs)

    def prepare(self, module, *args, example_inputs:list=None, example_kwargs:dict=None, sparsity_ratio=None, total_epochs=None, p=2.0, sparsity_global=False, copy_args=None,
            sparsity_type='n2m', sparsity_init_train_ep=5, sparsity_m=None, add_methods=True, aten_graph=True, copy_attrs=None, filter_func_register=None, weight_func_register=None,  transformation_dict=None, **kwargs):
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
        self.aten_graph = self.pre_dispatch = aten_graph
        
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
            sparsity_type=sparsity_type, sparsity_init_train_ep=sparsity_init_train_ep, sparsity_m=sparsity_m, add_methods=add_methods, aten_graph=aten_graph, copy_attrs=copy_attrs, filter_func_register=filter_func_register, weight_func_register=weight_func_register, transformation_dict=transformation_dict,**kwargs)

    #TODO pt2e implementation
    def get_layer_sparsity_ratio(self, sparsity_ratio=0.6):
        self.module = sparsity_func_wrapper.get_layer_sparsity_ratio(self.module, sparsity_ratio, transformation_dict=self.transformation_dict)
        return self
    

    def train(self, mode: bool = True,**kwargs): 
        # this super().train will call all submodules train() wich includes self.module
        # that will effectively call quant_func.train with self.module twice
        # to avoid that we directly set self.training
        # super().train(mode)
        self.training = mode

        if mode:
            # return quant_func.train(self.module, *args, **kwargs)
            self.module = sparsity_func_wrapper.train(self.module, mode, transformation_dict=self.transformation_dict, **kwargs)
        else:
            self.module = sparsity_func_wrapper.eval(self.module, mode, transformation_dict=self.transformation_dict, **kwargs)
        #
        return self
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
        
    def insert_parametrization(self, binary_mask=False):
        # for each of the nodes/layers, we calculate the parametrization/ mask and then register it over the weights and biases
        self.module = sparsity_func_wrapper.insert_parametrization(self.module, binary_mask=binary_mask, transformation_dict=self.transformation_dict)
        return self
    
    def remove_parametrization(self, leave_parameterized=True):
        # leave_parametrized=False would leave the original parameter instead of the parametrized parameter
        self.module = sparsity_func_wrapper.remove_parametrization(self.module, leave_parameterized=leave_parameterized, transformation_dict=self.transformation_dict)
        return self
    
    def calculate_sparsity(self):
        self.module = sparsity_func_wrapper.calculate_sparsity(self.module, transformation_dict=self.transformation_dict)
        return self
