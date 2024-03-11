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
from ... import xnn
from .utils import get_bn_adjusted_weight, create_bn_conv_mapping, create_next_conv_node_list, find_all_connected_conv, get_net_weight_node_channel_prune, get_net_weights_all,create_channel_pruned_model


class IncrementalPruningParametrization(nn.Module):
    # incrementally a portion of weights are completely zeroed out every epoch
    def __init__(self, curr_node, modules, channel_pruning=False, pruning_ratio=0.6, n2m_pruning=False,  
                 init_train_ep=5, net_weights = None, binary_mask=False, tao=1e-4, **kwargs):
        super().__init__()
        
        self.curr_node = curr_node
        self.all_modules = modules        
        self.channel_pruning = channel_pruning
        self.tao = tao
        self.binary_mask= binary_mask
        self.n2m_pruning = n2m_pruning
        self.pruning_ratio = pruning_ratio
        self.init_train_ep = init_train_ep
        self.fpgm_weights = True
        if "epoch_count" in kwargs:
            self.epoch_count = kwargs.get("epoch_count")
        if "total_epochs" in kwargs:
            self.total_epochs = kwargs.get("total_epochs")
        if "prunechannelunstructured" in kwargs:
            self.prunechannelunstructured = kwargs.get("prunechannelunstructured")
        if "m" in kwargs:
            self.m = kwargs.get("m")
            
        net_weight = net_weights[self.curr_node.target]
        
        # avoid error in backprop with detach
        net_weight = net_weight.detach()
        
        # TODO : some problem with dealing with the depthwise layer, something wrong in logic
        is_depthwise = (self.all_modules[self.curr_node.target].weight.shape[1] == 1) # we do not want to prune depthwise layers
        if int(self.pruning_ratio*net_weight.nelement())==0 or net_weight.size(0)<=32:
            if channel_pruning:
                mask = torch.ones(net_weight.size(0)).to(net_weight.device)
            else:
                mask = torch.ones_like(net_weight)
        elif is_depthwise and not(channel_pruning):
            mask = torch.ones_like(net_weight)
        else:
            mask = self.create_mask(net_weight)

        self.mask = mask
                
    def create_mask(self, net_weight):
        # epoch count is one indexed for some reason, manage according to that
        if self.epoch_count<=self.init_train_ep: # do not start pruning before the init train ep
            if self.channel_pruning:
                soft_mask = torch.ones(net_weight.size(0), device = net_weight.device)
            else:
                soft_mask = torch.ones_like(net_weight)
        else:
            # epoch by which network should be pruned as desired
            total_epochs_knee_point = (self.total_epochs-self.init_train_ep)*2//3 
            alpha_factor = 0
                     
            if self.n2m_pruning: 
                # self.m is the m in n:m pruning
                weight_abs = torch.abs(net_weight)
                soft_mask = torch.ones_like(net_weight)
                for i in range(math.ceil(len(net_weight.view(-1))/self.m)):
                    start_iter = self.m*i
                    end_iter = min(self.m*(i+1), len(net_weight.view(-1)))
                    prune_elements = (self.pruning_ratio)*(self.epoch_count-self.init_train_ep)/(total_epochs_knee_point-self.init_train_ep)
                    keep_elem_k = min(int((1-prune_elements)*self.m), end_iter - start_iter)
                    if (keep_elem_k==0) or ((end_iter - start_iter - keep_elem_k)==0):
                        continue    
                    Wh = torch.topk(torch.abs(weight_abs).view(-1)[start_iter:end_iter], k=keep_elem_k, largest=True)
                    Wl = torch.topk(torch.abs(weight_abs).view(-1)[start_iter:end_iter], k=(end_iter - start_iter - keep_elem_k), largest=False)
                    t = (torch.min(Wh.values)+ torch.max(Wl.values))/2
                    soft_mask.view(-1)[start_iter:end_iter] = (weight_abs.view(-1)[start_iter:end_iter] < t)*alpha_factor + (weight_abs.view(-1)[start_iter:end_iter] >= t)*1.0
            
            else:
                if self.channel_pruning:
                     # FPGM based finding channels to prune
                    if self.fpgm_weights:
                        weight_to_opt = net_weight
                        rough_median = torch.median(weight_to_opt, dim=0).values
                        all_dist = torch.ones(weight_to_opt.shape[0], device = net_weight.device)
                        for i in range(all_dist.shape[0]):
                            all_dist[i] = torch.norm(weight_to_opt[i]-rough_median, p=2)
                        net_weight=all_dist
                    else:
                        # L2 norm based finding channel to prune
                        L2_norm_channel = torch.ones(net_weight.size(0), device = net_weight.device)
                        for i in range(net_weight.size(0)):
                            L2_norm_channel[i] = torch.norm(net_weight[i], p=1).item() 
                        net_weight = L2_norm_channel
                
                # unstructured + channel pruning
                weight_abs = torch.abs(net_weight)
                prune_elements = (self.pruning_ratio)*(self.epoch_count-self.init_train_ep)/(total_epochs_knee_point-self.init_train_ep)
                keep_elem_k = int((1-prune_elements)*weight_abs.nelement())
                Wh = torch.topk(torch.abs(weight_abs).view(-1), k=keep_elem_k, largest=True)
                Wl = torch.topk(torch.abs(weight_abs).view(-1), k=(weight_abs.nelement() - keep_elem_k), largest=False)
                t = (torch.min(Wh.values)+ torch.max(Wl.values))/2
                soft_mask = (weight_abs < t)*alpha_factor + (weight_abs >= t)*1.0
        
        return soft_mask 
        
    def forward(self, X):
        if self.mask.device!=X.device: # temporary fix, not sure what the exact issue is, only needed for torchvision testing #TODO
            self.mask = self.mask.to(X.device)
            
        if self.channel_pruning: # pruning cannot be same for both weight and bias
            if len(X.shape) == 4: # weights 
                return X * self.mask[:,None,None,None]
            elif len(X.shape) == 1: # bias 
                return X * self.mask
        else: # X shape is of length 4 
            return X * self.mask

    def right_inverse(self, A):
        return A


class SoftPruningParametrization(nn.Module):
    # Parametrization technique where the weights are not completely zeroed out, however, they are pruned to zero incrementally with every epoch
    def __init__(self, curr_node, modules, channel_pruning=False, pruning_ratio=0.6, n2m_pruning=False,  
                 init_train_ep=5, net_weights = None, binary_mask=False, tao=1e-4, **kwargs):
        super().__init__()        
        
        # if there is batchnorm after the conv layer, then we will be using the net weight which is the combination of both batchnorm and conv to find the net weight
        
        self.curr_node = curr_node
        self.all_modules = modules        
        self.channel_pruning = channel_pruning
        self.tao = tao
        self.binary_mask= binary_mask
        self.n2m_pruning = n2m_pruning
        self.pruning_ratio = pruning_ratio
        self.init_train_ep = init_train_ep
        self.fpgm_weights = True
        if "epoch_count" in kwargs:
            self.epoch_count = kwargs.get("epoch_count")
        if "total_epochs" in kwargs:
            self.total_epochs = kwargs.get("total_epochs")
        if "prunechannelunstructured" in kwargs:
            self.prunechannelunstructured = kwargs.get("prunechannelunstructured")
        if "m" in kwargs:
            self.m = kwargs.get("m")
            
        net_weight = net_weights[self.curr_node.target]
        
        # avoid error in backprop with detach
        net_weight = net_weight.detach()
        
        # TODO : some problem with dealing with the depthwise layer, something wrong in logic
        is_depthwise = (self.all_modules[self.curr_node.target].weight.shape[1] == 1) # we do not want to prune depthwise layers
        if int(self.pruning_ratio*net_weight.nelement())==0 or is_depthwise: # do not prune layers with less than 32 channels only for channel pruning
            if channel_pruning:
                mask = torch.ones(net_weight.size(0)).to(net_weight.device)
            else:
                mask = torch.ones_like(net_weight).to(net_weight.device)
        elif net_weight.size(0)<=32 and channel_pruning:
            mask = torch.ones(net_weight.size(0)).to(net_weight.device)
        else:
            mask = self.create_mask(net_weight)

        if self.binary_mask:
            self.mask = (mask >= 0.5)
        else:
            self.mask = mask
                
    def create_mask(self, net_weight):
        return torch.ones_like(net_weight)
        
    def forward(self, X):
        if self.mask.device!=X.device: # temporary fix, not sure what the exact issue is, only needed for torchvision testing #TODO
            self.mask = self.mask.to(X.device)
            
        if self.channel_pruning: # pruning cannot be same for both weight and bias
            if len(X.shape) == 4: # weights
                return X * self.mask[:,None,None,None]
            elif len(X.shape) == 1: # bias 
                return X * self.mask
        else: # X shape is of length 4 
            return X * self.mask

    def right_inverse(self, A):
        return A


class SigmoidPruningParametrization(SoftPruningParametrization):
    # epoch count is one indexed for some reason, manage according to that
    def create_mask(self, net_weight):
        if self.epoch_count<=self.init_train_ep:
            if self.channel_pruning:
                soft_mask = torch.ones(net_weight.size(0), device = net_weight.device)
            else:
                soft_mask = torch.ones_like(net_weight)
                
        else:             
            if self.n2m_pruning:
                # prune n elements for every m elements (pass n/m in the self.pruning_ratio)
                weight_abs = torch.abs(net_weight)
                soft_mask = torch.ones_like(net_weight)
                for i in range(math.ceil(len(net_weight.view(-1))/self.m)):
                    start_iter = self.m*i
                    end_iter = min(self.m*(i+1), len(net_weight.view(-1)))
                    keep_elem_k = min(int((1-self.pruning_ratio)*self.m), end_iter - start_iter)
                    if (keep_elem_k==0) or ((end_iter - start_iter - keep_elem_k)==0):
                        continue
                    Wh = torch.topk(torch.abs(weight_abs).view(-1)[start_iter:end_iter], k=keep_elem_k, largest=True)
                    Wl = torch.topk(torch.abs(weight_abs).view(-1)[start_iter:end_iter], k=(end_iter - start_iter - keep_elem_k), largest=False)
                    t = (torch.min(Wh.values)+ torch.max(Wl.values))/2
                    weight_offseted.view(-1)[start_iter:end_iter] = weight_abs.view(-1)[start_iter:end_iter] - t

            else:
                if self.channel_pruning:
                     # FPGM based finding channels to prune
                    if self.fpgm_weights:
                        weight_to_opt = net_weight
                        rough_median = torch.median(weight_to_opt, dim=0).values
                        all_dist = torch.ones(weight_to_opt.shape[0], device = net_weight.device)
                        for i in range(all_dist.shape[0]):
                            all_dist[i] = torch.norm(weight_to_opt[i]-rough_median, p=2) #L2 distance
                        net_weight=all_dist
                    
                    # # channel pruning, calculating L2 norm for each channel and pruning on the basis of that
                    else:
                        L2_norm_channel = torch.ones(net_weight.size(0), device = net_weight.device)
                        for i in range(net_weight.size(0)):
                            L2_norm_channel[i] = torch.norm(net_weight[i], p=2).item()
                        net_weight = L2_norm_channel

                # unstructured pruning + channel pruning
                weight_abs = torch.abs(net_weight)
                keep_elem_k = int((1-self.pruning_ratio)*weight_abs.nelement())
                Wh = torch.topk(torch.abs(weight_abs).view(-1), k=keep_elem_k, largest=True)
                Wl = torch.topk(torch.abs(weight_abs).view(-1), k=(weight_abs.nelement() - keep_elem_k), largest=False)
                t = (torch.min(Wh.values)+ torch.max(Wl.values))/2
                weight_offseted = weight_abs - t
                
            sigmoid_domain = 3.0
            offset_factor = (2*sigmoid_domain+2)*(((self.epoch_count-self.init_train_ep))/self.total_epochs) #trying to push values away from 0, to find clear boundary/mask
            weight_spread = 1.0*(self.epoch_count-self.init_train_ep)/self.total_epochs  # limiting weights closer to 3 in the initial epochs to prevent huge pruning
            weight_min, weight_max = xnn.utils.extrema(weight_offseted, range_shrink_percentile=0.01)
            weight_offseted_abs_max = max(abs(weight_min), abs(weight_max))
            
            weight_offseted_scaled = (weight_offseted * weight_spread / weight_offseted_abs_max) + sigmoid_domain
            weight_offseted_scaled = weight_offseted_scaled + ((weight_offseted_scaled<sigmoid_domain)*(-offset_factor) + (weight_offseted_scaled>=sigmoid_domain)*(offset_factor))
            soft_mask = torch.nn.functional.hardsigmoid(weight_offseted_scaled) 
        # hard sigmoid has a limit, below -3, it is 0, above +3, it is 1. Between them it is (x+3)/6
        
        return soft_mask      
                 

class BlendPruningParametrization(SoftPruningParametrization):
    def __init__(self, curr_node, modules, channel_pruning=False, pruning_ratio=0.6, n2m_pruning=False, init_train_ep=5, net_weights=None, binary_mask=False, tao=0.0001, **kwargs):
        if 'p' in kwargs:
            p = kwargs.pop('p')
        else:
            p=None
        self.p =p
        super().__init__(curr_node, modules, channel_pruning, pruning_ratio, n2m_pruning, init_train_ep, net_weights, binary_mask, tao, **kwargs)
    def create_mask(self, net_weight):
        # epoch count is one indexed for some reason, manage according to that
        if self.epoch_count<=self.init_train_ep:
            if self.channel_pruning:
                soft_mask = torch.ones(net_weight.size(0), device = net_weight.device)
            else:
                soft_mask = torch.ones_like(net_weight)
        else:
            # alpha factor gets multiplied to weights that needs to be pruned, it starts with 1 and parabolically moves towards 0
            total_epochs_knee_point = (self.total_epochs-self.init_train_ep)*2//3
            if self.epoch_count<=self.init_train_ep:
                alpha_factor = 1
            elif self.epoch_count>total_epochs_knee_point:
                alpha_factor = 0
            else:
                alpha_factor = math.pow(abs(self.epoch_count-total_epochs_knee_point),self.p)/math.pow(total_epochs_knee_point-self.init_train_ep, self.p)
                     
            if self.n2m_pruning:
                # prune 41 elements for every 64 elements (pass 41/64 in the self.pruning_ratio)
                weight_abs = torch.abs(net_weight)
                soft_mask = torch.ones_like(net_weight)
                for i in range(math.ceil(len(net_weight.view(-1))/self.m)):
                    start_iter = self.m*i
                    end_iter = min(self.m*(i+1), len(net_weight.view(-1)))
                    keep_elem_k = min(int((1-self.pruning_ratio)*self.m), end_iter - start_iter)
                    if (keep_elem_k==0) or ((end_iter - start_iter - keep_elem_k)==0):
                        continue
                    Wh = torch.topk(torch.abs(weight_abs).view(-1)[start_iter:end_iter], k=keep_elem_k, largest=True)
                    Wl = torch.topk(torch.abs(weight_abs).view(-1)[start_iter:end_iter], k=(end_iter - start_iter - keep_elem_k), largest=False)
                    t = (torch.min(Wh.values)+ torch.max(Wl.values))/2
                    soft_mask.view(-1)[start_iter:end_iter] = (weight_abs.view(-1)[start_iter:end_iter] < t)*alpha_factor + (weight_abs.view(-1)[start_iter:end_iter] >= t)*1.0
            
            elif self.prunechannelunstructured:
                # prune the pruning ratio number of elements in each layer of the model instead of considering the weights of full model
                weight_abs = torch.abs(net_weight)
                soft_mask = torch.ones_like(net_weight)
                for i in range(weight_abs.shape[0]):
                    keep_elem_k = int((1-self.pruning_ratio)*weight_abs[i].nelement())
                    if keep_elem_k==0:
                        continue
                    Wh = torch.topk(torch.abs(weight_abs[i]).view(-1), k=keep_elem_k, largest=True)
                    Wl = torch.topk(torch.abs(weight_abs[i]).view(-1), k=(weight_abs[i].nelement() - keep_elem_k), largest=False)
                    t = (torch.min(Wh.values)+ torch.max(Wl.values))/2
                    soft_mask[i] = (weight_abs[i] < t)*alpha_factor + (weight_abs[i] >= t)*1.0
            
            else:
                if self.channel_pruning:
                     # FPGM based finding channels to prune
                    if self.fpgm_weights:
                        weight_to_opt = net_weight
                        rough_median = torch.median(weight_to_opt, dim=0).values
                        all_dist = torch.ones(weight_to_opt.shape[0], device = net_weight.device)
                        for i in range(all_dist.shape[0]):
                            all_dist[i] = torch.norm(weight_to_opt[i]-rough_median, p=2)
                        net_weight=all_dist
                        
                    else:
                        # L2 norm based finding channel to prune
                        L2_norm_channel = torch.ones(net_weight.size(0), device = net_weight.device)
                        for i in range(net_weight.size(0)):
                            L2_norm_channel[i] = torch.norm(net_weight[i], p=2).item() 
                        net_weight = L2_norm_channel
                
                # unstructured + channel pruning
                weight_abs = torch.abs(net_weight)
                keep_elem_k = int((1-self.pruning_ratio)*weight_abs.nelement())
                Wh = torch.topk(torch.abs(weight_abs).view(-1), k=keep_elem_k, largest=True)
                Wl = torch.topk(torch.abs(weight_abs).view(-1), k=(weight_abs.nelement() - keep_elem_k), largest=False)
                t = (torch.min(Wh.values)+ torch.max(Wl.values))/2
                soft_mask = (weight_abs < t)*alpha_factor + (weight_abs >= t)*1.0
                
        return soft_mask   
    
         
class PrunerModule(torch.nn.Module):
    def __init__(self, module, pruning_ratio=None, total_epochs=None, pruning_class='blend',p=2.0, copy_args=[],
                 pruning_global=False, pruning_type='channel', pruning_init_train_ep=5, pruning_m=None, **kwargs) -> None:
        super().__init__(**kwargs)
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
            
        pruning_class_dict = {"blend": BlendPruningParametrization, 
                 "sigmoid": SigmoidPruningParametrization, 
                 "incremental": IncrementalPruningParametrization}
        
        self.pruning_class = pruning_class_dict[pruning_class]
        
        #responsible for creating a next mapping (basically helps combine the weight of BN and conv)
        self.next_bn_nodes = create_bn_conv_mapping(module)
        
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
        
        if self.channel_pruning:
            # creating the next node list, which contains the connection to all convs to the current conv
            self.next_conv_node_list = create_next_conv_node_list(module)
            # returns the list of all conv that share the same output
            self.all_connected_nodes = find_all_connected_conv(module)
        else:
            self.next_conv_node_list = None
            self.all_connected_nodes = None
        
        if self.n2m_pruning and self.global_pruning:
            print("Cannot do both global pruning along with n2m pruning, it doesn't make sense! \n")
            raise NotImplementedError
        
        for copy_arg in copy_args:
            setattr(self, copy_arg, getattr(module, copy_arg))
            
        # to get net weights for each of the layers, incorporating all the required dependancies
        self.net_weights = get_net_weights_all(module, self.next_conv_node_list, self.all_connected_nodes, self.next_bn_nodes, self.channel_pruning, self.global_pruning)
        
        if self.global_pruning:
            if self.channel_pruning:
                self.get_layer_pruning_ratio_channel(pruning_ratio)
            else:
                self.get_layer_pruning_ratio(pruning_ratio)
        #
    
    def get_layer_pruning_ratio(self, pruning_ratio=0.6):
        fx_model = fx.symbolic_trace(self.module)
        modules = dict(fx_model.named_modules())
        model_graph = fx_model.graph
        # can also include the batchnorm merged weights over here
        set_of_all_weights = torch.empty(0)
        for node in model_graph.nodes:
            if node.target=='output':
                continue
            if node.args and isinstance(node.target, str):
                if isinstance(modules[node.target], torch.nn.Conv2d) and (modules[node.target].weight.size(1) != 1):
                    set_of_all_weights = torch.cat((set_of_all_weights, modules[node.target].weight.mean(dim=[1,2,3]))) if self.channel_pruning else torch.cat((set_of_all_weights, modules[node.target].weight.view(-1)))
                    
        topk = torch.topk(torch.abs(set_of_all_weights), k=int(pruning_ratio*len(set_of_all_weights)), largest=False)
        indexes = topk.indices
        sorted_idx, _ = torch.sort(indexes)
        
        pruning_ratio = dict()
        idx_iter = 0
        total_params = 0
        limit_factor=0.7 # to not prune a layer with more than limit_factor*100% of its weights
        for node in model_graph.nodes:
            if node.target=='output':
                continue
            if node.args and isinstance(node.target, str):
                if isinstance(modules[node.target], torch.nn.Conv2d):
                    if (modules[node.target].weight.size(1) != 1):
                        net_params = modules[node.target].weight.shape[0] if self.channel_pruning else torch.numel(modules[node.target].weight)
                        total_params+=net_params
                        curr_idx = torch.min((sorted_idx < total_params),0).indices.item()-1
                        if curr_idx<0:
                            curr_idx = len(sorted_idx)
                        pruning_ratio[node.target] = (curr_idx-idx_iter)/net_params
                        if pruning_ratio[node.target] >= limit_factor:
                            print("{} is getting the pruning ratio of {} which is a problem, so limiting to {}".format(node.target, pruning_ratio[node.target], limit_factor))
                            pruning_ratio[node.target] = limit_factor
                        idx_iter = curr_idx
                    else:
                        pruning_ratio[node.target] = 0 
                    
        self.pruning_ratio = pruning_ratio
        print(pruning_ratio)
      
        return self
    
    def get_layer_pruning_ratio_channel(self, pruning_ratio=0.6): ################## not complete TODO : Global channel pruning
        fx_model = fx.symbolic_trace(self.module)
        modules = dict(fx_model.named_modules())
        model_graph = fx_model.graph
        # can also include the batchnorm merged weights over here
        # set_of_all_weights = torch.empty(0)
        set_of_all_channel_avg_norms = torch.empty(0)
        for node in model_graph.nodes:
            if node.target=='output':
                continue
            if node.args and isinstance(node.target, str):
                if isinstance(modules[node.target], torch.nn.Conv2d):
                    if (modules[node.target].weight.shape[1]!=1) and (modules[node.target].weight.shape[0]>32):
                        mean, std = torch.mean(self.net_weights[node.target]), torch.std(self.net_weights[node.target])
                        k = (self.net_weights[node.target]-mean)/std
                        set_of_all_channel_avg_norms = torch.cat((set_of_all_channel_avg_norms, k))
                    # set_of_all_weights = torch.cat((set_of_all_weights, modules[node.target].weight.view(-1)))
                    
        topk = torch.topk(torch.abs(set_of_all_channel_avg_norms), k=int(len(set_of_all_channel_avg_norms)*pruning_ratio), largest=False)
        indexes = topk.indices
        sorted_idx, _ = torch.sort(indexes)
        
        pruning_ratio = dict()
        idx_iter = 0
        total_params = 0
        max_prune = 0.8 # to not prune a layer with more than max_prune*100% of its channels
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
                pruning_ratio[node.target] = min((curr_idx-idx_iter)/net_params, max_prune)
                idx_iter = curr_idx 
        
        # depthwise is not pruning same as the previous conv, maybe some where, 1-2 extra channels pruned, will reduce accuracy           
        self.pruning_ratio = pruning_ratio
        print(pruning_ratio)
      
        return self
      
    def train(self, mode: bool = True): 
        super().train(mode)
        if mode: #train mode
            self.remove_parametrization(leave_parameterized=False) # we do not want to keep the old mask, rest of the weights are adjusted according to this one
            self.epoch_count += 1
            self.insert_parametrization()
            
        elif self.epoch_count==self.total_epochs: # evaluation in the final epoch, we would want to completely prune out the weights
            self.insert_parametrization(binary_mask=True) # binary_mask=True gives hard mask
            self.remove_parametrization()
            self.calculate_sparsity()
            print("The final sparsity of the network is {}".format(self.sparsity))
        return self
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
        
    def insert_parametrization(self, binary_mask=False):
        # for each of the nodes/layers, we calculate the parametrization/ mask and then register it over the weights and biases
        
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
                    pruning_ratio = self.pruning_ratio if isinstance(self.pruning_ratio, float) else self.pruning_ratio[node.target]
                    if self.pruning_class == BlendPruningParametrization:
                        p_kwargs = {'p':self.p}
                    else:
                        p_kwargs = {}
                    
                    parameterization = self.pruning_class(curr_node=node, modules=modules, channel_pruning=self.channel_pruning, pruning_ratio=pruning_ratio,
                                                        n2m_pruning=self.n2m_pruning, init_train_ep=self.init_train_ep, prunechannelunstructured=self.prunechannelunstructured,
                                                        epoch_count=self.epoch_count, total_epochs=self.total_epochs, net_weights = self.net_weights, binary_mask=binary_mask, m=self.m,**p_kwargs)
                    
                    parametrize.register_parametrization(modules[node.target], "weight", parameterization)
                    if self.channel_pruning:
                        if modules[node.target].bias is not None:
                            parametrize.register_parametrization(modules[node.target], "bias", parameterization)
                        if node.target in self.next_bn_nodes:
                            parametrize.register_parametrization(modules[self.next_bn_nodes[node.target].target], "weight", parameterization) 
                            parametrize.register_parametrization(modules[self.next_bn_nodes[node.target].target], "bias", parameterization)
                        # also put the same parametrization in the next dwconv(along with its BN), if there is one connected to this 
                        for n_id in self.next_conv_node_list[node.target]:
                            if modules[n_id.target].weight.shape[1]==1: #n_id node is dwconv 
                                parametrize.register_parametrization(modules[n_id.target], "weight", parameterization)
                                if modules[n_id.target].bias is not None:
                                    parametrize.register_parametrization(modules[n_id.target], "bias", parameterization)
                                if n_id.target in self.next_bn_nodes:
                                    parametrize.register_parametrization(modules[self.next_bn_nodes[n_id.target].target], "weight", parameterization) 
                                    parametrize.register_parametrization(modules[self.next_bn_nodes[n_id.target].target], "bias", parameterization)
                        
        return self
    
    def remove_parametrization(self, leave_parameterized=True):
        # leave_parametrized=False would leave the original parameter instead of the parametrized parameter
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
        
        # Make layer wise computionally pruning ratio, overall as well #TODO  

        for module_name, module in self.module.named_modules():
            
            if isinstance(module, torch.nn.Linear):
                continue
            
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