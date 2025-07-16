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

try:
    from torchvision import models as tvmodels
    has_tv = True
except Exception as e:
    has_tv = False
import torch
import torch.nn as nn
import torch.fx as fx
import torch.nn.utils.parametrize as parametrize
from torch.ao.quantization import quantize_fx
import copy

try:
    from timm import models as tmmodels
    has_timm = True
except:
    has_timm = False

from ...utils.optimization_base import OptimizationBaseModule
from ...utils.transformation_utils import wrapped_transformation_fn


from .utils import get_bn_adjusted_weight, create_bn_conv_mapping, create_next_conv_node_list, find_all_connected_nodes, get_net_weight_node_channel_prune, get_net_weights_all,create_channel_pruned_model,_call_functions_to_look
from .parametrization import BlendPruningParametrization, SigmoidPruningParametrization, IncrementalPruningParametrization, HeadChannelBlendPruningParametrization, HeadOnlyBlendPruningParametrization, ChannelOnlyBlendPruningParametrization, PRUNING_CLASS_DICT
from . import pruning_func_wrapper
class PrunerModule(OptimizationBaseModule): 
    def __init__(self, module, *args, pruning_ratio=None, total_epochs=None, pruning_class='blend',p=2.0, pruning_global=False, copy_args=None,
                 pruning_type='channel', pruning_init_train_ep=5, pruning_m=None, add_methods=True, transformation_dict=None, copy_attrs=None, **kwargs) -> None:
        copy_attrs = copy_attrs or []
        copy_args = copy_args or []
        super().__init__( module, *args, transformation_dict=transformation_dict, copy_attrs=copy_attrs, **kwargs)
        self.prepare(module, *args, pruning_ratio=pruning_ratio, total_epochs=total_epochs, pruning_class=pruning_class, copy_args = copy_args,
                    p=p,pruning_global=pruning_global, pruning_type=pruning_type, pruning_init_train_ep=pruning_init_train_ep, pruning_m=pruning_m, add_methods=add_methods, transformation_dict=transformation_dict, **kwargs)

    def prepare(self, module, *args, pruning_ratio=None, total_epochs=None, pruning_class='blend',p=2.0, pruning_global=False, copy_args=None,
                pruning_type='channel', pruning_init_train_ep=5, pruning_m=None, add_methods=True, transformation_dict=None, **kwargs):
        copy_args = copy_args or []
        self.module = module
        
        self.epoch_count = 0
        self.pruning_ratio = pruning_ratio
        self.total_epochs = total_epochs
        self.sparsity = 0
        self.init_train_ep = pruning_init_train_ep
        self.p = p
        
        if pruning_ratio==0:
            raise RuntimeError("pruning ratio of 0 is not supported , try turning off pruning and trying again")
        if not(pruning_ratio and total_epochs):
            raise RuntimeError("pruning ratio and total epochs are necessary to be provided")
        elif not(pruning_ratio):
            raise RuntimeError("pruning ratio should be provided")
        elif not(total_epochs):
            raise RuntimeError("total epochs should be provided")
        
        self.pruning_class = PRUNING_CLASS_DICT[pruning_class]
        
        #responsible for creating a next mapping (basically helps combine the weight of BN and conv)
        # self.next_bn_nodes = create_bn_conv_mapping(module)
    
        self.channel_pruning = False
        self.n2m_pruning = False
        self.prunechannelunstructured = False
        
        if pruning_type=='channel':
            self.channel_pruning = True
        elif pruning_type=='n2m':
            self.n2m_pruning = True
        elif pruning_type=='prunechannelunstructured':
            self.prunechannelunstructured = True
        elif pruning_type=='unstructured':
            pass
        self.global_pruning = pruning_global
        
        if self.n2m_pruning:
            if pruning_m is None:
                raise RuntimeError("The value of m should be provided in case of n:m pruning")
            else:
                self.m = pruning_m
        else:
            self.m = None
        
        # if self.channel_pruning:
        #     # creating the next node list, which contains the connection to all convs to the current conv
        #     self.next_conv_node_list = create_next_conv_node_list(module)
        #     # returns the list of all conv that share the same output
        #     self.all_connected_nodes = find_all_connected_nodes(module)
        # else:
        #     self.next_conv_node_list = None
        #     self.all_connected_nodes = None
        
        if self.n2m_pruning and self.global_pruning:
            print("Cannot do both global pruning along with n2m pruning, it doesn't make sense! \n")
            raise NotImplementedError
        
        for copy_arg in copy_args:
            setattr(self, copy_arg, getattr(module, copy_arg))
            
        # to get net weights for each of the layers, incorporating all the required dependancies
        # self.net_weights = get_net_weights_all(module, self.next_conv_node_list, self.all_connected_nodes, self.next_bn_nodes, self.channel_pruning, self.global_pruning)
        
        # if self.global_pruning:
        #     if self.channel_pruning:
        #         self.get_layer_pruning_ratio_channel(pruning_ratio)
        #     else:
        #         self.get_layer_pruning_ratio(pruning_ratio)
        #
        self.module = pruning_func_wrapper.init(self.module, *args, pruning_ratio=pruning_ratio, total_epochs=total_epochs, pruning_class=pruning_class, copy_args = copy_args,
                    p=p,pruning_global=pruning_global, pruning_type=pruning_type, pruning_init_train_ep=pruning_init_train_ep, pruning_m=pruning_m, add_methods=add_methods, transformation_dict=transformation_dict, **kwargs)
    
    def get_layer_pruning_ratio(self, pruning_ratio=0.6):
        self.module = pruning_func_wrapper.get_layer_pruning_ratio(self.module, pruning_ratio, transformation_dict=self.transformation_dict)
        return self
    
    def get_layer_pruning_ratio_channel(self, pruning_ratio=0.6): ################## not complete TODO : Global channel pruning
        self.module = pruning_func_wrapper.get_layer_pruning_ratio_channel(self.module, pruning_ratio, transformation_dict=self.transformation_dict)
        return self

    def train(self, mode: bool = True): 
        self. module = pruning_func_wrapper.train(self.module, mode=mode, transformation_dict=self.transformation_dict)
        return self
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
        
    def insert_parametrization(self, binary_mask=False):
        # for each of the nodes/layers, we calculate the parametrization/ mask and then register it over the weights and biases
        self.module = pruning_func_wrapper.insert_parametrization(self.module, binary_mask=binary_mask, transformation_dict=self.transformation_dict)
        return self
    
    def remove_parametrization(self, leave_parameterized=True):
        # leave_parametrized=False would leave the original parameter instead of the parametrized parameter
        self.module = pruning_func_wrapper.remove_parametrization(self.module, leave_parameterized=leave_parameterized, transformation_dict=self.transformation_dict)
        return self
           
    def calculate_sparsity(self):
        self.module = pruning_func_wrapper.calculate_sparsity(self.module, transformation_dict=self.transformation_dict)
        return self


class PrunerQuantModule(PrunerModule): # still under development
    def __init__(self, module, pruning_ratio=0.8, total_epochs=10, pruning_class='blend', copy_args=[], quant_backend='qnnpack', 
                 pruning_global=False, pruning_type='channel', pruning_init_train_ep=5, **kwargs) -> None:
        super().__init__(module, pruning_ratio, total_epochs, pruning_class, copy_args, pruning_global, pruning_type, pruning_init_train_ep, **kwargs)
        
        self.module = nn.Sequential(
            torch.ao.quantization.QuantStub(),
            module,
            torch.ao.quantization.DeQuantStub()
        )
        self.example_inputs = [torch.randn(1,3,224,224)]
        self.qconfig_mapping = torch.ao.quantization.get_default_qat_qconfig_mapping(quant_backend)
        self.module = quantize_fx.prepare_qat_fx(self.module, self.qconfig_mapping, self.example_inputs) 
        
    def train(self, mode: bool = True):
        super().train(mode)

        if self.epoch_count == self.total_epochs and not mode:
            print("Now convert the fake quantized model to quantized one")
            self.convert()
                
        return self

    def convert(self, inplace=False, device='cpu'):
        self.module_quant = copy.deepcopy(self.module)
        self.module_quant = self.module_quant.to(torch.device(device))
        self.module_quant = quantize_fx.convert_fx(self.module_quant)
        self.module_quant.to(torch.device(next(self.module.parameters()).device))
        return self
    
