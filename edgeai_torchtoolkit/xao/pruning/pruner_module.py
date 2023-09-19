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
import torch.nn.utils.parametrize as parametrize
from torch.ao.quantization import quantize_fx
import copy
import math
import enum


class PruningParametrization(nn.Module):
    def __init__(self, module1, module2=None, channel_pruning=False, pruning_ratio=0.2, **kwargs):
        super().__init__()
        
        # if there is batchnorm after the conv layer, then we will be using the net weight which is the combination of both batchnorm and conv to find the net weight
        
        self.channel_pruning = channel_pruning
        
        if module2!=None: #might only work for conv right now, need to see to it
            target = module1.weight.shape
            bn_weight = module2.weight[:, None, None, None].expand(target)
            bn_var = torch.sqrt((module2.running_var + module2.eps)[:, None, None, None].expand(target))
            net_weight = torch.div(torch.mul(module1.weight, bn_weight), bn_var)  
            # 4-dim output
        else:
            net_weight = module1.weight 
        
        if self.channel_pruning:
            topk = torch.topk(torch.abs(net_weight).mean(dim=[1,2,3]), k=int(pruning_ratio*net_weight.size(0)), largest=False)
            mask = torch.ones(net_weight.size(0), device = net_weight.device)
            mask[topk.indices] = 0
            self.mask = mask
        else:
            topk = torch.topk(torch.abs(net_weight).view(-1), k=int(pruning_ratio*net_weight.nelement()), largest=False)
            mask = torch.ones_like(net_weight)
            mask.view(-1)[topk.indices] = 0
            self.mask = mask

    def forward(self, X):
        if self.channel_pruning:
            if len(X.shape) == 4:
                return X * self.mask[:,None,None,None]
            elif len(X.shape) == 1:
                return X * self.mask
        else:
            return X * self.mask

    def right_inverse(self, A):
        return A

class PDPPruningParametrization(nn.Module):
    def __init__(self, module1, module2=None, channel_pruning=False, pruning_ratio=0.2, tao=1e-4, binary_mask=False, n2m_pruning=False, **kwargs):
        super().__init__()        
        
        # if there is batchnorm after the conv layer, then we will be using the net weight which is the combination of both batchnorm and conv to find the net weight
        
        self.channel_pruning = channel_pruning
        self.tao = tao
        self.binary_mask= binary_mask
        self.n2m_pruning = n2m_pruning
        
        if module2!=None: #might only work for conv right now, need to see to it
            target = module1.weight.shape
            bn_weight = module2.weight[:, None, None, None].expand(target)
            bn_var = torch.sqrt((module2.running_var + module2.eps)[:, None, None, None].expand(target))
            net_weight = torch.div(torch.mul(module1.weight.clone(), bn_weight), bn_var)
            # 4-dim output
        else:
            net_weight = module1.weight.clone()
        
        if self.channel_pruning:
            print("Not yet implemented")
            raise NotImplementedError
        
        elif self.n2m_pruning:
            # prune 41 elements for every 64 elements
            mask = torch.ones_like(net_weight)
            net_weight2 = torch.pow(net_weight,2)/self.tao
            # exp of values were going inf and because of that nan was coming, limiting that
            net_weight2 = torch.where(net_weight2>80, 80, net_weight2)
            
            if (int((1-pruning_ratio)*64)!=0) and (int(pruning_ratio*64)!=0):
                for i in range(math.ceil(len(net_weight.view(-1))/64)):
                    start_iter = 64*i
                    end_iter = min(64*(i+1), len(mask.view(-1)))
                    Wh = torch.topk(torch.abs(net_weight).view(-1)[start_iter:end_iter], k=int((1-pruning_ratio)*64), largest=True)
                    Wl = torch.topk(torch.abs(net_weight).view(-1)[start_iter:end_iter], k=int(pruning_ratio*64), largest=False)
                    t = (torch.min(Wh.values)+ torch.max(Wl.values))/2
                    t2 = torch.pow(t,2)/self.tao
                    
                    mask.view(-1)[start_iter:end_iter] = torch.exp(net_weight2.view(-1)[start_iter:end_iter]) / (torch.exp(net_weight2.view(-1)[start_iter:end_iter]) +  torch.exp(t2))
                
            mask.detach_()
                
            if self.binary_mask:
                self.mask = (mask >= 0.5)
            else:
                self.mask = mask
            
        else:
            if int(pruning_ratio*net_weight.nelement())==0 or int((1-pruning_ratio)*net_weight.nelement())==0:
                self.mask = torch.ones_like(net_weight)
                
            else:
                Wh = torch.topk(torch.abs(net_weight).view(-1), k=int((1-pruning_ratio)*net_weight.nelement()), largest=True)
                Wl = torch.topk(torch.abs(net_weight).view(-1), k=int(pruning_ratio*net_weight.nelement()), largest=False)
                t = (torch.min(Wh.values)+ torch.max(Wl.values))/2
                t2 = torch.pow(t,2)/self.tao
                net_weight2 = torch.pow(net_weight,2)/self.tao
                # exp of values were going inf and because of that nan was coming, limiting that
                net_weight2 = torch.where(net_weight2>80, 80, net_weight2)
                mask = torch.exp(net_weight2) / (torch.exp(net_weight2) +  torch.exp(t2))
                mask.detach_()
                if self.binary_mask:
                    self.mask = (mask >= 0.5)
                else:
                    self.mask = mask
            
    def forward(self, X):
        if self.channel_pruning:
            if len(X.shape) == 4:
                return X * self.mask[:,None,None,None]
            elif len(X.shape) == 1:
                return X * self.mask
        else:
            return X * self.mask

    def right_inverse(self, A):
        return A


class PrunerModule(torch.nn.Module):
    def __init__(self, module, pruning_ratio=0.5, total_epochs=10, pruning_class=PruningParametrization,
                 pruning_type=None, channel_pruning=False, copy_args=[],
                 train_epoch_per_iter=5, epsilon=0.015, global_pruning=False, n2m_pruning=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module = module
        
        self.epoch_count = 0
        self.pruning_ratio = pruning_ratio
        self.total_epochs = total_epochs
        self.pruning_class = pruning_class
        self.sparsity = 0
        self.train_epoch_per_iter = train_epoch_per_iter
        self.epsilon = epsilon
        
        self.next_module_names = dict()
        self.create_mapping()
        
        self.channel_pruning = channel_pruning
        self.global_pruning = global_pruning
        self.n2m_pruning = n2m_pruning
        
        if self.global_pruning:
            self.get_layer_pruning_ratio()
        
        for copy_arg in copy_args:
            setattr(self, copy_arg, getattr(module, copy_arg))
        #
    
    def get_layer_pruning_ratio(self):
        fx_model = fx.symbolic_trace(self.module)
        modules = dict(fx_model.named_modules())
        model_graph = fx_model.graph
        # can also include the batchnorm merged weights over here
        pruning_ratio = dict()
        set_of_all_weights = torch.empty(0)
        for node in model_graph.nodes:
            if node.target=='output':
                continue
            if node.args and isinstance(node.target, str):
                if isinstance(modules[node.target], torch.nn.Conv2d):
                    set_of_all_weights = torch.cat((set_of_all_weights, modules[node.target].weight.view(-1)))
                    
        topk = torch.topk(torch.abs(set_of_all_weights), k=int(0.6*len(set_of_all_weights)), largest=False)
        indexes = topk.indices
        sorted_idx, _ = torch.sort(indexes)
        
        idx_iter = 0
        total_params = 0
        for node in model_graph.nodes:
            if node.target=='output':
                continue
            if node.args and isinstance(node.target, str):
                if isinstance(modules[node.target], torch.nn.Conv2d):
                    net_params = torch.numel(modules[node.target].weight)
                    total_params+=net_params
                    curr_idx = torch.min((sorted_idx < total_params),0).indices.item()-1
                    if curr_idx<0:
                        curr_idx = len(sorted_idx)
                    pruning_ratio[node.target] = (curr_idx-idx_iter)/net_params
                    idx_iter = curr_idx 
                    
        self.pruning_ratio = pruning_ratio
      
        return self
        
    
    def create_mapping(self):
        # need to check all the layers connected to conv2d and linear and we would have to include them all in the mapping list
        
        if isinstance(self.module, torch.fx.GraphModule): # the QAT model is already a graph and thus cannnot be traced
            fx_model = self.module
        else:
            fx_model = fx.symbolic_trace(self.module)
        
        # the QAT module will already be merged and thus we would not have to calculate all this.
            
        modules = dict(fx_model.named_modules())
        model_graph = fx_model.graph
        for node in model_graph.nodes:
            if node.target=='output':
                continue
            if node.args and isinstance(node.target, str) and (node.target in modules):
                if isinstance(modules[node.target], torch.nn.BatchNorm2d):
                    if len(node.args[0].users)>1: # dont merge if conv has multiple users
                        continue
                    if isinstance(modules[node.args[0].target], torch.nn.Conv2d):
                    # if isinstance(modules[node.args[0].target], torch.nn.Conv2d) or isinstance(modules[node.args[0].target], torch.nn.Linear):
                        self.next_module_names[node.args[0].target] = node.target
      
    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.remove_parametrization(leave_parameterized=False)
            self.epoch_count += 1
            self.insert_parametrization()
            
        elif self.epoch_count==self.total_epochs:
            self.insert_parametrization(binary_mask=True)
            self.remove_parametrization()
            self.calculate_sparsity()
            
        # else:  # not of use probably, it will be nothing (0)
        #     self.calculate_sparsity()

        return self
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
        
    def insert_parametrization(self, binary_mask=False):
        # start_epoch=max(round(self.total_epochs*0.15),0)
        # last_epoch = (self.total_epochs-start_epoch)
        if self.n2m_pruning: # hard coded this for our use case, however need to make the code changeable
            self.pruning_ratio = 41/64
            
        if self.pruning_class == PDPPruningParametrization:
            if isinstance(self.pruning_ratio, float):
                curr_pruning_ratio =  self.pruning_ratio*min(1, self.epsilon*(self.epoch_count))
            else: # layer level pruning
                curr_pruning_ratio = copy.deepcopy(self.pruning_ratio)
                curr_pruning_ratio.update((x, y*min(1, self.epsilon*(self.epoch_count))) for x, y in self.pruning_ratio.items())
                
        else:
            if isinstance(self.pruning_ratio, float):
                curr_pruning_ratio = self.pruning_ratio*((self.epoch_count//self.train_epoch_per_iter)/(self.total_epochs//self.train_epoch_per_iter))
            else:
                curr_pruning_ratio = copy.deepcopy(self.pruning_ratio)
                curr_pruning_ratio.update((x, y*((self.epoch_count//self.train_epoch_per_iter)/(self.total_epochs//self.train_epoch_per_iter))) for x, y in self.pruning_ratio.items())
                
        # pruning_ratio = max(self.pruning_ratio*(self.epoch_count-start_epoch)/(last_epoch-start_epoch),0)
        # pruning_ratio = min(pruning_ratio, self.pruning_ratio)
        
        if isinstance(self.module, torch.fx.GraphModule): # the QAT model is already a graph and thus cannnot be traced
            fx_model = self.module
        else:
            fx_model = fx.symbolic_trace(self.module)
            
        modules = dict(fx_model.named_modules())
        model_graph = fx_model.graph
        
        for node in model_graph.nodes:
            if node.target=='output':
                continue
            if node.args and isinstance(node.target, str) and (node.target in modules):
                if isinstance(modules[node.target], nn.Conv2d):
                # if isinstance(modules[node.target], nn.Conv2d) or isinstance(modules[node.target], nn.Linear):
                    if node.target in self.next_module_names:
                        # send both the modules(conv/fc + BN) to the pruning
                        if isinstance(self.pruning_ratio, float):
                            parameterization = self.pruning_class(modules[node.target], modules[self.next_module_names[node.target]], self.channel_pruning, curr_pruning_ratio, binary_mask=binary_mask, n2m_pruning=self.n2m_pruning)
                        else:
                            parameterization = self.pruning_class(modules[node.target], modules[self.next_module_names[node.target]], self.channel_pruning, curr_pruning_ratio[node.target], binary_mask=binary_mask)
                        
                        parametrize.register_parametrization(modules[node.target], "weight", parameterization)
                        if self.channel_pruning:
                            if modules[node.target].bias is not None:
                                parametrize.register_parametrization(modules[node.target], "bias", parameterization)
                            parametrize.register_parametrization(modules[self.next_module_names[node.target]], "weight", parameterization) 
                            parametrize.register_parametrization(modules[self.next_module_names[node.target]], "bias", parameterization)      
                        #
                                                 
                    else:
                        # send only the original one 
                        if isinstance(self.pruning_ratio, float):
                            parameterization = self.pruning_class(modules[node.target], None, self.channel_pruning, curr_pruning_ratio, binary_mask=binary_mask, n2m_pruning=self.n2m_pruning)
                        else:
                            parameterization = self.pruning_class(modules[node.target], None, self.channel_pruning, curr_pruning_ratio[node.target], binary_mask=binary_mask)
                        
                        parametrize.register_parametrization(modules[node.target], "weight", parameterization)
                        if self.channel_pruning:
                            if modules[node.target].bias is not None:
                                parametrize.register_parametrization(modules[node.target], "bias", parameterization) 

        return self
    
    def remove_parametrization(self, leave_parameterized=True):
        for module_name, module in self.module.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
                if parametrize.is_parametrized(module, 'weight'):
                    parametrize.remove_parametrizations(module, "weight", leave_parametrized=leave_parameterized) 
                if parametrize.is_parametrized(module, 'bias'):
                    parametrize.remove_parametrizations(module, "bias", leave_parametrized=leave_parameterized)
        return self
           
    def calculate_sparsity(self):
        num_zeros = 0
        num_elements = 0

        for module_name, module in self.module.named_modules():
            
            if isinstance(module, (torch.ao.nn.quantized.Conv2d, torch.ao.nn.quantized.Linear)):
                num_zeros += torch.sum(module.weight()==0).item()
                num_zeros += torch.sum(module.bias()==0).item()
                num_elements += torch.numel(module.weight())
                num_elements += torch.numel(module.bias())
                
            elif isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                for param_name, param in module.named_parameters():
                    if "weight" in param_name:
                        num_zeros += torch.sum(param == 0).item()
                        num_elements += param.nelement()
                    if "bias" in param_name:
                        num_zeros += torch.sum(param == 0).item()
                        num_elements += param.nelement()
                
            elif hasattr(module, 'weight'):
                num_elements += torch.numel(module.weight)
                num_zeros += (torch.numel(module.weight) - torch.count_nonzero(module.weight).item())
            
                if hasattr(module, 'bias'):
                    num_elements += torch.numel(module.bias)
                    num_zeros += (torch.numel(module.bias) - torch.count_nonzero(module.bias).item())
            
        self.sparsity = num_zeros / num_elements
        return self

class _QuantExptModule(torch.nn.Module):
    def __init__(self, module, total_epochs=10, quant_backend='qnnpack', copy_args=[], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.module = nn.Sequential(
            torch.ao.quantization.QuantStub(),
            module,
            torch.ao.quantization.DeQuantStub()
        )
        self.example_inputs = [torch.randn(1,3,224,224)]
        self.qconfig_mapping = torch.ao.quantization.get_default_qat_qconfig_mapping(quant_backend)
        
        self.epoch_count = 0
        self.total_epochs = total_epochs
        
        for copy_arg in copy_args:
            setattr(self, copy_arg, getattr(module, copy_arg))
        #
        
    def train(self, mode: bool = True):
        super().train(mode)
        if self.epoch_count == 0:
            self.module = quantize_fx.prepare_qat_fx(self.module, self.qconfig_mapping, self.example_inputs)
            
        if mode:
            self.epoch_count += 1
            
        elif self.epoch_count == self.total_epochs:
            self.convert()
                
        return self
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
                
    def convert(self, inplace=False, device='cpu'):
        self.module_quant = copy.deepcopy(self.module)
        self.module_quant = self.module_quant.to(torch.device(device))
        self.module_quant = quantize_fx.convert_fx(self.module_quant, inplace=inplace)
        return self

class PrunerQuantModule(PrunerModule):
    def __init__(self, module, pruning_ratio=0.5, total_epochs=10, pruning_class=PruningParametrization, quant_backend='qnnpack', channel_pruning=False, copy_args=[], train_epoch_per_iter=5, *args, **kwargs) -> None:
        super().__init__(module, pruning_ratio, total_epochs, pruning_class, channel_pruning, copy_args,train_epoch_per_iter, *args, **kwargs)
        
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

        if self.epoch_count == self.total_epochs:
            self.convert()
                
        return self

    def convert(self, inplace=False, device='cpu'):
        self.module_quant = copy.deepcopy(self.module)
        self.module_quant = self.module_quant.to(torch.device(device))
        self.module_quant = quantize_fx.convert_fx(self.module_quant)
        return self
