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


from ast import mod
from importlib.util import module_from_spec
from re import S
from edgeai_torchmodelopt.xmodelopt import pruning
from numpy import source
from timm import models as tmmodels
from torchvision import models as tvmodels
import torch
from torch import _dynamo as torch_dynamo
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.utils.source_matcher_utils import  SourcePartition
from torch.fx.passes.utils.matcher_utils import InternalMatch
import torch.nn.utils.parametrize as parametrize
from torch.ao.quantization import quantize_fx
import copy
import math
from .... import xnn
from .utils import get_bn_adjusted_weight, create_bn_conv_mapping, create_next_conv_node_list, find_all_connected_nodes, get_net_weight_node_channel_prune, get_net_weights_all,create_channel_pruned_model,get_pruning_partitions,_call_functions_to_look


def is_depthwise(fx_model,source,source_partition):
    if source not in [nn.Conv2d]:
        return False
    FINAL_EXCEPTION = Exception(f'Still not Supported for {source}' and {type(source_partition)})
    if source == nn.Conv2d:
        if isinstance(source_partition, SourcePartition):
            weight_node = source_partition.nodes[0]
        elif isinstance(source_partition,InternalMatch):
            weight_node = source_partition.nodes_map['_param_constant0']
        else:
            raise FINAL_EXCEPTION
    else:
        raise FINAL_EXCEPTION
    weight = getattr(fx_model,weight_node.target)
    return weight.shape[1]==1


class IncrementalPruningParametrization(nn.Module):
    # incrementally a portion of weights are completely zeroed out every epoch
    def __init__(self,fx_model, source, source_partition, channel_pruning=False, pruning_ratio=0.6, n2m_pruning=False,  
                 init_train_ep=5, net_weight = None, binary_mask=False, tao=1e-4, pruning_dim=0, **kwargs):
        super().__init__()
        
        # self.fx_model = fx_model
        self.source = source
        self.source_partition = source_partition        
        self.channel_pruning = channel_pruning
        self.tao = tao
        self.binary_mask= binary_mask
        self.n2m_pruning = n2m_pruning
        self.pruning_ratio = pruning_ratio
        self.init_train_ep = init_train_ep
        self.fpgm_weights = True
        self.pruning_dim = pruning_dim
        if "epoch_count" in kwargs:
            self.epoch_count = kwargs.get("epoch_count")
        if "total_epochs" in kwargs:
            self.total_epochs = kwargs.get("total_epochs")
        if "prunechannelunstructured" in kwargs:
            self.prunechannelunstructured = kwargs.get("prunechannelunstructured")
        if "m" in kwargs:
            self.m = kwargs.get("m")
            
        self.new_shape = [s for s in range(len(net_weight.shape))]
        self.new_shape[0],self.new_shape[pruning_dim] = self.new_shape[pruning_dim],self.new_shape[0]
        # avoid error in backprop with detach
        net_weight = net_weight.detach()
        
        # TODO : some problem with dealing with the depthwise layer, something wrong in logic
        
        # we do not want to prune depthwise layers
        if int(self.pruning_ratio*net_weight.nelement())==0 or net_weight.size(self.pruning_dim)<=32:
            if channel_pruning:
                mask = torch.ones(net_weight.size(self.pruning_dim)).to(net_weight.device)
            else:
                mask = torch.ones_like(net_weight)
        elif is_depthwise(fx_model,source,source_partition) and not(channel_pruning):
            mask = torch.ones_like(net_weight)
        else:
            mask = self.create_mask(net_weight)

        self.mask = mask
                
    def create_mask(self, net_weight):
        # epoch count is one indexed for some reason, manage according to that
        if self.epoch_count<=self.init_train_ep: # do not start pruning before the init train ep
            if self.channel_pruning:
                soft_mask = torch.ones(net_weight.size(self.pruning_dim), device = net_weight.device)
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
                    net_weight = torch.permute(net_weight,self.new_shape)
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
            if len(X.shape) == 1: # bias 
                return X * self.mask
            mask_shape = list(self.mask.shape)
            while len(mask_shape) < len(X.shape):
                new_shape = mask_shape.copy()
                new_shape.append(1)
                mask_shape = new_shape
            if self.pruning_dim ==0:
                if len(X.shape) == 4: # weights
                    return X * self.mask[:,None,None,None]
                else:
                    return X * self.mask.reshape(mask_shape)
            else:
                shape= [s for s in range(len(X.shape))]
                shape[0],shape[self.pruning_dim] = shape[self.pruning_dim],shape[0]
                X = X.permute(shape)
                result = X*self.mask.reshape(mask_shape)
                return result.permute(shape)
        else: # X shape is of length 4 
            return X * self.mask

    def right_inverse(self, A):
        return A
    def __repr__(self):
        
        return f'{self.__class__.__name__}(...)'

class SoftPruningParametrization(nn.Module):
    # Parametrization technique where the weights are not completely zeroed out, however, they are pruned to zero incrementally with every epoch
    def __init__(self, fx_model, source, source_partition, channel_pruning=False, pruning_ratio=0.6, n2m_pruning=False,  init_train_ep=5, net_weight = None, binary_mask=False, tao=1e-4, pruning_dim = 0, **kwargs):
        super().__init__()        
        
        # if there is batchnorm after the conv layer, then we will be using the net weight which is the combination of both batchnorm and conv to find the net weight
        
        # self.fx_model = fx_model
        self.source = source
        self.source_partition = source_partition        
        self.channel_pruning = channel_pruning
        self.tao = tao
        self.binary_mask= binary_mask
        self.n2m_pruning = n2m_pruning
        self.pruning_ratio = pruning_ratio
        self.init_train_ep = init_train_ep
        self.fpgm_weights = True
        self.pruning_dim = pruning_dim
        if "epoch_count" in kwargs:
            self.epoch_count = kwargs.get("epoch_count")
        if "total_epochs" in kwargs:
            self.total_epochs = kwargs.get("total_epochs")
        if "prunechannelunstructured" in kwargs:
            self.prunechannelunstructured = kwargs.get("prunechannelunstructured")
        if "m" in kwargs:
            self.m = kwargs.get("m")
            
        self.new_shape = [s for s in range(len(net_weight.shape))]
        self.new_shape[0],self.new_shape[pruning_dim] = self.new_shape[pruning_dim],self.new_shape[0]
        # avoid error in backprop with detach
        net_weight = net_weight.detach()
        
        # TODO : some problem with dealing with the depthwise layer, something wrong in logic
         # we do not want to prune depthwise layers
        if int(self.pruning_ratio*net_weight.nelement())==0 or is_depthwise(fx_model,source,source_partition): # do not prune layers with less than 32 channels only for channel pruning
            if channel_pruning:
                mask = torch.ones(net_weight.size(self.pruning_dim)).to(net_weight.device)
            else:
                mask = torch.ones_like(net_weight).to(net_weight.device)
        elif net_weight.size(self.pruning_dim)<=32 and channel_pruning:
            mask = torch.ones(net_weight.size(self.pruning_dim)).to(net_weight.device)
        else:
            mask = self.create_mask(net_weight)

        if self.binary_mask:
            self.mask = (mask >= 0.5)
        else:
            self.mask = mask
                
    def create_mask(self, net_weight):
        return torch.ones_like(net_weight)
        
    def forward(self, X:torch.Tensor):
        if self.mask.device!=X.device: # temporary fix, not sure what the exact issue is, only needed for torchvision testing #TODO
            self.mask = self.mask.to(X.device)
            
        if self.channel_pruning: # pruning cannot be same for both weight and bias
            if len(X.shape) == 1: # bias 
                return X * self.mask
            mask_shape = list(self.mask.shape)
            while len(mask_shape) < len(X.shape):
                new_shape = mask_shape.copy()
                new_shape.append(1)
                mask_shape = new_shape
            if self.pruning_dim ==0:
                if len(X.shape) == 4: # weights
                    return X * self.mask[:,None,None,None]
                else:
                    return X * self.mask.reshape(mask_shape)
            else:
                shape= [s for s in range(len(X.shape))]
                shape[0],shape[self.pruning_dim] = shape[self.pruning_dim],shape[0]
                X = X.permute(shape)
                result = X*self.mask.reshape(mask_shape)
                return result.permute(shape)
        else: # X shape is of length 4 
            return X * self.mask

    def right_inverse(self, A:torch.Tensor):
        return A

    def __repr__(self):
        return f'{self.__class__.__name__}(...)'

class SigmoidPruningParametrization(SoftPruningParametrization):
    # epoch count is one indexed for some reason, manage according to that
    
    def create_mask(self, net_weight):
        if self.epoch_count<=self.init_train_ep:
            if self.channel_pruning:
                soft_mask = torch.ones(net_weight.size(self.pruning_dim), device = net_weight.device)
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
                    net_weight = torch.permute(net_weight,self.new_shape)
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
    def __init__(self, fx_model,source,source_partition, channel_pruning=False, pruning_ratio=0.6, n2m_pruning=False, init_train_ep=5, net_weight=None, binary_mask=False, tao=0.0001,p=2,pruning_dim =0, **kwargs):
        self.p =p
        super().__init__(fx_model, source,source_partition, channel_pruning, pruning_ratio, n2m_pruning, init_train_ep, net_weight, binary_mask, tao, pruning_dim=pruning_dim, **kwargs)
    def create_mask(self, net_weight):
        # epoch count is one indexed for some reason, manage according to that
        if self.epoch_count<=self.init_train_ep:
            if self.channel_pruning:
                soft_mask = torch.ones(net_weight.size(self.pruning_dim), device = net_weight.device)
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
                for i in range(weight_abs.shape[self.pruning_dim]):
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
                    net_weight = torch.permute(net_weight,self.new_shape)
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


def get_num_heads_head_dims(fx_model,source,source_partition):
    # TODO proper implementation for all attention layers with their variant
    attn_layers = [nn.MultiheadAttention,
                      tvmodels.swin_transformer.ShiftedWindowAttention,
                      tvmodels.swin_transformer.shifted_window_attention,
                      tmmodels.vision_transformer.Attention,
                      tmmodels.swin_transformer.WindowAttention]
    if source not in attn_layers:
        raise Exception(f'This is only for attention layer of transformer models so expected any of\n {",".join(attn_layers)}\n as source but got {source}')
    FINAL_EXCEPTION = Exception(f'Still not Supported for {source}' and {type(source_partition)})
    if source == nn.MultiheadAttention:
        if isinstance(source_partition,SourcePartition):
            if len(source_partition.nodes) ==  52:
                i = 30
            if len(source_partition.nodes) == 43:
                i = 23
            elif len(source_partition.nodes) == 60:
                i = 28     
                    
            shape = source_partition.nodes[i].args[1]
            num_heads,head_dims =  shape[1],shape[3]
        else: 
            raise FINAL_EXCEPTION
    else: 
        raise FINAL_EXCEPTION
            
    return num_heads,head_dims


#Experimental to check either only Channel Pruning works for  MultiHeadAttention Layer's inner_projection weights
class ChannelOnlyBlendPruningParametrization(BlendPruningParametrization):
    def __init__(self, fx_model,source,source_partition, channel_pruning=False, pruning_ratio=0.6, n2m_pruning=False, init_train_ep=5, net_weight=None, binary_mask=False, tao=0.0001, p=2,pruning_dim = 0, **kwargs):
        self.old_shape = list(net_weight.shape)
        assert len(self.old_shape) == 2, 'This parametrization is only for \'MultiHeadAttention\' Layer\'s in_proj weight and bias. \n So, weight of the node given must have dimension of 2'
        assert  self.old_shape[0]%self.old_shape[1] == 0, 'The number of heads should be obtainable from weights i.e., weight.shape[0] must be divisible by weight.shape[1] '
        self.num_heads , self.head_dim =get_num_heads_head_dims(fx_model,source,source_partition)
        self.shape1 =[3,self.num_heads,self.head_dim,self.old_shape[1]]
        super().__init__(fx_model,source,source_partition, channel_pruning, pruning_ratio, n2m_pruning, init_train_ep, net_weight, binary_mask, tao, p,pruning_dim, **kwargs)
        
    def create_mask(self, net_weight):
        if self.epoch_count<=self.init_train_ep:
            if self.channel_pruning:
                soft_mask = torch.ones(net_weight.size(self.pruning_dim), device = net_weight.device)
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
                for i in range(weight_abs.shape[self.pruning_dim]):
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
                    
                    net_weight=torch.reshape(net_weight,self.shape1)
                    #channel Pruning
                    net_weight = net_weight.permute(2,0,1,3)
                    if self.fpgm_weights:
                        weight_to_opt = net_weight
                        rough_median = torch.median(weight_to_opt, dim=0).values
                        all_dist = torch.ones(weight_to_opt.shape[0], device = net_weight.device)
                        for i in range(all_dist.shape[0]):
                            all_dist[i] = torch.norm(weight_to_opt[i]-rough_median, p=2)
                        channel_norm = all_dist                    
                    else:
                        # L2 norm based finding channel to prune
                        L2_norm_channel = torch.ones(net_weight.size(0), device = net_weight.device)
                        for i in range(net_weight.size(0)):
                            L2_norm_channel[i] = torch.norm(net_weight[i], p=2).item() 
                        channel_norm = L2_norm_channel 
                    
                    weight_abs = torch.abs(channel_norm)
                    soft_mask = torch.ones(self.shape1[:3])
                    soft_mask = soft_mask.permute(2,1,0)
                    keep_elem_k = int((1-self.pruning_ratio)*weight_abs.nelement())
                    Wh = torch.topk(torch.abs(weight_abs).view(-1), k=keep_elem_k, largest=True)
                    Wl = torch.topk(torch.abs(weight_abs).view(-1), k=(weight_abs.nelement() - keep_elem_k), largest=False)
                    t = (torch.min(Wh.values)+ torch.max(Wl.values))/2
                    for i, w in enumerate(weight_abs):
                        if w < t:
                            soft_mask[i] *= alpha_factor
                    soft_mask = soft_mask.permute(2,1,0)
                    soft_mask = soft_mask.reshape(-1)
                
                else:
                    weight_abs = torch.abs(net_weight)
                    keep_elem_k = int((1-self.pruning_ratio)*weight_abs.nelement())
                    Wh = torch.topk(torch.abs(weight_abs).view(-1), k=keep_elem_k, largest=True)
                    Wl = torch.topk(torch.abs(weight_abs).view(-1), k=(weight_abs.nelement() - keep_elem_k), largest=False)
                    t = (torch.min(Wh.values)+ torch.max(Wl.values))/2
                    soft_mask = (weight_abs < t)*alpha_factor + (weight_abs >= t)*1.0

        return soft_mask 


#Experimental to check either only Head Pruning works for MultiHeadAttention Layer's inner_projection weights
class HeadOnlyBlendPruningParametrization(BlendPruningParametrization):
    def __init__(self, fx_model,source,source_partition, channel_pruning=False, pruning_ratio=0.6, n2m_pruning=False, init_train_ep=5, net_weight=None, binary_mask=False, tao=0.0001, p=2, pruning_dim = 0, **kwargs):
        self.old_shape = list(net_weight.shape)
        assert len(self.old_shape) == 2, 'This parametrization is only for \'MultiHeadAttention\' Layer\'s in_proj weight and bias. \n So, weight of the node given must have dimension of 2'
        assert  self.old_shape[0]%self.old_shape[1] == 0, 'The number of heads should be obtainable from weights i.e., weight.shape[0] must be divisible by weight.shape[1] '
        self.num_heads , self.head_dim =get_num_heads_head_dims(fx_model,source,source_partition)
        self.shape1 =[3,self.num_heads,self.head_dim,self.old_shape[1]]
        self.shape1 =[3,self.num_heads,self.old_shape[0]//(3*self.num_heads),self.old_shape[1]]
        super().__init__(fx_model,source,source_partition, channel_pruning, pruning_ratio, n2m_pruning, init_train_ep, net_weight, binary_mask, tao, p,pruning_dim, **kwargs)
        
    def create_mask(self, net_weight):
        if self.epoch_count<=self.init_train_ep:
            if self.channel_pruning:
                soft_mask = torch.ones(net_weight.size(self.pruning_dim), device = net_weight.device)
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
                for i in range(weight_abs.shape[self.pruning_dim]):
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
                    
                    net_weight=torch.reshape(net_weight,self.shape1)
                    #head Pruning
                    net_weight = net_weight.permute(1,0,2,3)
                    if self.fpgm_weights:
                        weight_to_opt = net_weight
                        rough_median = torch.median(weight_to_opt, dim=0).values
                        all_dist = torch.ones(weight_to_opt.shape[0], device = net_weight.device)
                        for i in range(all_dist.shape[0]):
                            all_dist[i] = torch.norm(weight_to_opt[i]-rough_median, p=2)
                        head_norm = all_dist                    
                    else:
                        # L2 norm based finding channel to prune
                        L2_norm_channel = torch.ones(net_weight.size(0), device = net_weight.device)
                        for i in range(net_weight.size(0)):
                            L2_norm_channel[i] = torch.norm(net_weight[i], p=2).item() 
                        head_norm = L2_norm_channel 
                    
                    weight_abs = torch.abs(head_norm)
                    soft_mask = torch.ones(self.shape1[:3])
                    soft_mask = soft_mask.permute(1,0,2)
                    keep_elem_k = int((1-self.pruning_ratio)*weight_abs.nelement())
                    Wh = torch.topk(torch.abs(weight_abs).view(-1), k=keep_elem_k, largest=True)
                    Wl = torch.topk(torch.abs(weight_abs).view(-1), k=(weight_abs.nelement() - keep_elem_k), largest=False)
                    t = (torch.min(Wh.values)+ torch.max(Wl.values))/2
                    for i, w in enumerate(weight_abs):
                        if w < t:
                            soft_mask[i] *= alpha_factor
                    soft_mask = soft_mask.permute(1,0,2)
                    soft_mask = soft_mask.reshape(-1)
                
                else:
                    weight_abs = torch.abs(net_weight)
                    keep_elem_k = int((1-self.pruning_ratio)*weight_abs.nelement())
                    Wh = torch.topk(torch.abs(weight_abs).view(-1), k=keep_elem_k, largest=True)
                    Wl = torch.topk(torch.abs(weight_abs).view(-1), k=(weight_abs.nelement() - keep_elem_k), largest=False)
                    t = (torch.min(Wh.values)+ torch.max(Wl.values))/2
                    soft_mask = (weight_abs < t)*alpha_factor + (weight_abs >= t)*1.0

        return soft_mask 


class HeadChannelBlendPruningParametrization(BlendPruningParametrization):
    def __init__(self, fx_model,source,source_partition, channel_pruning=False, pruning_ratio=0.6, n2m_pruning=False, init_train_ep=5, net_weight=None, binary_mask=False, tao=0.0001, p=2, pruning_dim = 0, **kwargs):
        self.old_shape = list(net_weight.shape)
        assert len(self.old_shape) == 2, 'This parametrization is only for \'MultiHeadAttention\' Layer\'s in_proj weight and bias. \n So, weight of the node given must have dimension of 2'
        assert  self.old_shape[0]%self.old_shape[1] == 0, 'The number of heads should be obtainable from weights i.e., weight.shape[0] must be divisible by weight.shape[1] '
        self.num_heads , self.head_dim =get_num_heads_head_dims(fx_model,source,source_partition)
        self.shape1 =[3,self.num_heads,self.head_dim,self.old_shape[1]]
        super().__init__(fx_model,source,source_partition, channel_pruning, pruning_ratio, n2m_pruning, init_train_ep, net_weight, binary_mask, tao, p,pruning_dim, **kwargs)
        
    def create_mask(self, net_weight):
        if self.epoch_count<=self.init_train_ep:
            if self.channel_pruning:
                soft_mask = torch.ones(net_weight.size(self.pruning_dim), device = net_weight.device)
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
                for i in range(weight_abs.shape[self.pruning_dim]):
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
                    
                    net_weight=torch.reshape(net_weight,self.shape1)
                    #channel Pruning
                    net_weight = net_weight.permute(2,0,1,3)
                    if self.fpgm_weights:
                        weight_to_opt = net_weight
                        rough_median = torch.median(weight_to_opt, dim=0).values
                        all_dist = torch.ones(weight_to_opt.shape[0], device = net_weight.device)
                        for i in range(all_dist.shape[0]):
                            all_dist[i] = torch.norm(weight_to_opt[i]-rough_median, p=2)
                        channel_norm = all_dist                    
                    else:
                        # L2 norm based finding channel to prune
                        L2_norm_channel = torch.ones(net_weight.size(0), device = net_weight.device)
                        for i in range(net_weight.size(0)):
                            L2_norm_channel[i] = torch.norm(net_weight[i], p=2).item() 
                        channel_norm = L2_norm_channel 
                    
                    weight_abs = torch.abs(channel_norm)
                    soft_mask = torch.ones(self.shape1[:3])
                    soft_mask = soft_mask.permute(2,1,0)
                    keep_elem_k = int((1-self.pruning_ratio)*weight_abs.nelement())
                    Wh = torch.topk(torch.abs(weight_abs).view(-1), k=keep_elem_k, largest=True)
                    Wl = torch.topk(torch.abs(weight_abs).view(-1), k=(weight_abs.nelement() - keep_elem_k), largest=False)
                    t = (torch.min(Wh.values)+ torch.max(Wl.values))/2
                    for i, w in enumerate(weight_abs):
                        if w < t:
                            soft_mask[i] *= alpha_factor
                    soft_mask = soft_mask.permute(2,1,0)
                    channel_soft_mask = soft_mask.reshape(-1)
                    
                    
                    #head Pruning
                    net_weight = net_weight.permute(2,1,0,3)
                    if self.fpgm_weights:
                        weight_to_opt = net_weight
                        rough_median = torch.median(weight_to_opt, dim=0).values
                        all_dist = torch.ones(weight_to_opt.shape[0], device = net_weight.device)
                        for i in range(all_dist.shape[0]):
                            all_dist[i] = torch.norm(weight_to_opt[i]-rough_median, p=2)
                        head_norm = all_dist                    
                    else:
                        # L2 norm based finding channel to prune
                        L2_norm_channel = torch.ones(net_weight.size(0), device = net_weight.device)
                        for i in range(net_weight.size(0)):
                            L2_norm_channel[i] = torch.norm(net_weight[i], p=2).item() 
                        head_norm = L2_norm_channel 
                    
                    weight_abs = torch.abs(head_norm)
                    soft_mask = torch.ones(self.shape1[:3])
                    soft_mask = soft_mask.permute(1,0,2)
                    keep_elem_k = int((1-self.pruning_ratio)*weight_abs.nelement())
                    Wh = torch.topk(torch.abs(weight_abs).view(-1), k=keep_elem_k, largest=True)
                    Wl = torch.topk(torch.abs(weight_abs).view(-1), k=(weight_abs.nelement() - keep_elem_k), largest=False)
                    t = (torch.min(Wh.values)+ torch.max(Wl.values))/2
                    for i, w in enumerate(weight_abs):
                        if w < t:
                            soft_mask[i] *= alpha_factor
                    soft_mask = soft_mask.permute(1,0,2)
                    head_soft_mask = soft_mask.reshape(-1)
                    
                    for i, w in enumerate(channel_soft_mask):
                        if w == 1:
                            channel_soft_mask[i] *= head_soft_mask[i]
                    soft_mask = channel_soft_mask
                
                else:
                    weight_abs = torch.abs(net_weight)
                    keep_elem_k = int((1-self.pruning_ratio)*weight_abs.nelement())
                    Wh = torch.topk(torch.abs(weight_abs).view(-1), k=keep_elem_k, largest=True)
                    Wl = torch.topk(torch.abs(weight_abs).view(-1), k=(weight_abs.nelement() - keep_elem_k), largest=False)
                    t = (torch.min(Wh.values)+ torch.max(Wl.values))/2
                    soft_mask = (weight_abs < t)*alpha_factor + (weight_abs >= t)*1.0

        return soft_mask 


class PrunerModule(torch.nn.Module):
    def __init__(self, module, example_args:list,example_kwargs:dict,pruning_ratio=None, total_epochs=None, pruning_class='blend',p=2.0, copy_args=[],
                 pruning_global=False, pruning_type='channel', pruning_init_train_ep=5, pruning_m=None, **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(module,fx.GraphModule):
            #TODO Differnetiate between fx and pt2e graph modules
            # Assuming Default pt2e here
            self.module = module
        else:
            self.module ,_= torch_dynamo.export(module,aten_graph=True)(*example_args,**example_kwargs)
            # with module.graph.inserting_after():
            #     for node in self.module.graph.nodes:
            #         if node.op != 'output' and len(node.users) == 0:
            #             self.module.graph.erase_node(node)
            # self.module.graph.lint(), self.module.recompile()
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
        
        self.pruning_partitions = get_pruning_partitions(self.module)
        self.next_bn_nodes = create_bn_conv_mapping(self.module,self.pruning_partitions)
        self.channel_pruning = False
        self.n2m_pruning = False
        self.prunechannelunstructured = False
        self.parametrized_params = set()
        
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
            self.next_conv_node_list = create_next_conv_node_list(self.module,self.pruning_partitions)
            # returns the list of all conv that share the same output
            self.all_connected_nodes = find_all_connected_nodes(self.module,self.pruning_partitions)
        else:
            self.next_conv_node_list = None
            self.all_connected_nodes = None
        
        if self.n2m_pruning and self.global_pruning:
            print("Cannot do both global pruning along with n2m pruning, it doesn't make sense! \n")
            raise NotImplementedError
        
        for copy_arg in copy_args:
            setattr(self, copy_arg, getattr(module, copy_arg))
            
        # to get net weights for each of the layers, incorporating all the required dependancies
        self.net_weights = get_net_weights_all(self.module,self.pruning_partitions,self.next_conv_node_list,self.all_connected_nodes,self.next_bn_nodes,self.channel_pruning,self.global_pruning)
        
        if self.global_pruning:
            if self.channel_pruning:
                self.get_layer_pruning_ratio_channel(pruning_ratio)
            else:
                self.get_layer_pruning_ratio(pruning_ratio)
        #

    #TODO pt2e implementation
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

    #TODO pt2e implementaion
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
                        net_weight ,dim = self.net_weights[node.name]
                        mean, std = torch.mean(net_weight,dim), torch.std(net_weight,dim)
                        k = (net_weight-mean)/std
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
        
        params = dict(self.module.named_parameters())
        
        # TODO Experiment among (HeadBlendPruningParametrization, HeadOnlyBlendPruningParametrization, ChannelOnlyBlendPruningParametrization)
        attn_proj_class = HeadChannelBlendPruningParametrization
        attn_proj_class = HeadOnlyBlendPruningParametrization
        attn_proj_class = ChannelOnlyBlendPruningParametrization
        
        all_partition_nodes =[] 
        for cls,partitions in self.pruning_partitions.items():
            for partition in partitions:
                nodes = []
                if isinstance(partition,SourcePartition): nodes.extend(partition.nodes)
                # if isinstance(partition,InternalMatch): nodes.extend(list(partition.nodes_map.values()))
                all_partition_nodes.extend([node.name for node in nodes])
        pruning_ratio = self.pruning_ratio if isinstance(self.pruning_ratio, float) else self.pruning_ratio[node.target]
        kwargs ={
            'channel_pruning' :self.channel_pruning,
            'pruning_ratio':pruning_ratio,
            'n2m_pruning':self.n2m_pruning, 
            'init_train_ep':self.init_train_ep, 
            'prunechannelunstructured':self.prunechannelunstructured, 
            'epoch_count':self.epoch_count, 
            'total_epochs':self.total_epochs, 
             'binary_mask':binary_mask, 
             'm':self.m,
            }
        if self.pruning_class == BlendPruningParametrization:
            kwargs['p'] = self.p
        for node in self.module.graph.nodes:
            if node.name in all_partition_nodes:
                continue
            if node.target  not in self.net_weights:
                continue 
            elif node.op == 'get_attr':
                attr = params[node.target]
                if isinstance(attr,nn.Parameter):
                    net_weight, dim = self.net_weights[node.target]
                    parametrization = self.pruning_class(fx_model=self.module,source=node,source_partition=node, net_weight = net_weight,pruning_dim = dim,**kwargs)
                    parametrize.register_parametrization(self.module, node.target, parametrization)
        
        True           
        for cls,partitions in self.pruning_partitions.items():
            if cls == nn.Conv2d:
                assert isinstance(partitions[0],SourcePartition)
                for partition in partitions:
                    param_name = partition.nodes[0].target
                    net_weight,dim =  self.net_weights[param_name]
                    parametrization = self.pruning_class(fx_model=self.module,source=cls,source_partition=partition, net_weight = net_weight,pruning_dim = dim,**kwargs)
                    parametrize.register_parametrization(self.module, param_name, parametrization)
                    if len(partition.nodes) ==3 and self.channel_pruning:
                        param_name = partition.nodes[1].target
                        parametrize.register_parametrization(self.module, param_name, parametrization)
                    
            elif cls == nn.BatchNorm2d:
                assert isinstance(partitions[0],SourcePartition)
                for partition in partitions:
                    param_name = partition.nodes[3].target
                    net_weight,dim =  self.net_weights[param_name]
                    parametrization = self.pruning_class(fx_model=self.module,source=cls,source_partition=partition, net_weight = net_weight,pruning_dim = dim,**kwargs)
                    parametrize.register_parametrization(self.module, param_name, parametrization)
                    if len(partition.nodes) ==11 and self.channel_pruning:
                        param_name = partition.nodes[4].target
                        parametrize.register_parametrization(self.module, param_name, parametrization)
            elif cls == nn.Linear:
                assert isinstance(partitions[0],SourcePartition)
                for partition in partitions:
                    if any( 'output' in [n.op for n in out.users]for out in partition.output_nodes):
                        continue
                    if len(partition.nodes) == 4:
                        i= 0
                        j=2
                    elif len(partition.nodes) == 6:
                        i= 1
                        j= 3
                    elif len(partition.nodes) == 8:
                        i= 3
                        j= 5
                    elif len(partition.nodes) == 7:
                        i= 2
                        j= 4
                    param_name = partition.nodes[i].target
                    net_weight,dim =  self.net_weights[param_name]
                    parametrization = self.pruning_class(fx_model=self.module,source=cls,source_partition=partition, net_weight = net_weight,pruning_dim = dim,**kwargs)
                    parametrize.register_parametrization(self.module, param_name, parametrization)
                    if len(partition.nodes) in (4,6) and self.channel_pruning:
                        param_name = partition.nodes[j].target
                        parametrize.register_parametrization(self.module, param_name, parametrization)
            elif cls == nn.LayerNorm:
                assert isinstance(partitions[0],SourcePartition)
                for partition in partitions:
                    param_name = partition.nodes[0].target
                    net_weight,dim =  self.net_weights[param_name]
                    parametrization = self.pruning_class(fx_model=self.module,source=cls,source_partition=partition, net_weight = net_weight,pruning_dim = dim,**kwargs)
                    parametrize.register_parametrization(self.module, param_name, parametrization)
                    if len(partition.nodes) ==6 and self.channel_pruning:
                        param_name = partition.nodes[1].target
                        parametrize.register_parametrization(self.module, param_name, parametrization)
            elif cls == nn.MultiheadAttention:
                assert isinstance(partitions[0],SourcePartition)
                for partition in partitions:
                    if len(partition.nodes) ==43:
                        i= 2
                    elif len(partition.nodes) == 52:
                        i = 1  
                    elif len(partition.nodes) == 60:
                        i = 1    
                    param_name = partition.nodes[i].target
                    net_weight,dim =  self.net_weights[param_name]
                    if self.channel_pruning:
                        parametrization1 = attn_proj_class(fx_model=self.module,source=cls,source_partition=partition, net_weight = net_weight,**kwargs)
                    else: 
                        parametrization1 = self.pruning_class(fx_model=self.module,source=cls,source_partition=partition, net_weight = net_weight,pruning_dim = dim,**kwargs)
                    parametrize.register_parametrization(self.module, param_name, parametrization1)
                    if len(partition.nodes) ==43:
                        i= 37
                    elif len(partition.nodes) == 52:
                        i = 46        
                    elif len(partition.nodes) == 60:
                        i = 54
                    param_name = partition.nodes[i].target
                    net_weight,dim =  self.net_weights[param_name]
                    parametrization2 = self.pruning_class(fx_model=self.module,source=cls,source_partition=partition, net_weight = net_weight,pruning_dim = dim,**kwargs)
                    parametrize.register_parametrization(self.module, param_name, parametrization2)
                    
                    if len(partition.nodes) ==43 and self.channel_pruning:
                        param_name = partition.nodes[4].target
                        parametrize.register_parametrization(self.module, param_name, parametrization1)
                        param_name = partition.nodes[39].target
                        parametrize.register_parametrization(self.module, param_name, parametrization2)
                    if len(partition.nodes) ==52 and self.channel_pruning:
                        param_name = partition.nodes[8].target
                        parametrize.register_parametrization(self.module, param_name, parametrization1)
                        param_name = partition.nodes[48].target
                        parametrize.register_parametrization(self.module, param_name, parametrization2)
                    if len(partition.nodes) ==60 and self.channel_pruning:
                        param_name = partition.nodes[8].target
                        parametrize.register_parametrization(self.module, param_name, parametrization1)
                        param_name = partition.nodes[56].target
                        parametrize.register_parametrization(self.module, param_name, parametrization2)
                                      
        return self
    
    def remove_parametrization(self, leave_parameterized=True):
        # leave_parametrized=False would leave the original parameter instead of the parametrized parameter
        params  = dict (self.module.named_parameters()) 
        
        for name,param in params.items():
            names = name.split('.')
            if len(names)>=3 and names[-1] == 'original' and names[-3] == 'parametrizations':
                module,param_name = self.module,names[-2]
                if parametrize.is_parametrized(module, param_name):
                    self.parametrized_params.add(param_name)
                    parametrize.remove_parametrizations(module, param_name, leave_parametrized=leave_parameterized) 
        return self
           
    def calculate_sparsity(self):
        num_zeros = 0
        num_elements = 0
        
        # Make layer wise computionally pruning ratio, overall as well #TODO 
        
        for param_name in self.parametrized_params:
            tensor = getattr(self.module,param_name)
            num_zeros += torch.sum(tensor==0).item()
            num_elements += torch.numel(tensor)
   
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
    
