try:
    from torchvision import models as tvmodels
    has_tv = True
except Exception as e:
    has_tv = False
import torch
import torch.nn as nn
import math

try:
    from timm import models as tmmodels
    has_timm = True
except:
    has_timm = False

from .... import xnn


class IncrementalPruningParametrization(nn.Module):
    # incrementally a portion of weights are completely zeroed out every epoch
    def __init__(self, curr_node, modules, channel_pruning=False, pruning_ratio=0.6, n2m_pruning=False,  
                 init_train_ep=5, net_weight = None, binary_mask=False, tao=1e-4, pruning_dim=0, **kwargs):
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
        is_depthwise = (self.curr_node.target in self.all_modules) and (self.all_modules[self.curr_node.target].weight.shape[1] == 1) # we do not want to prune depthwise layers
        if int(self.pruning_ratio*net_weight.nelement())==0 or net_weight.size(self.pruning_dim)<=32:
            if channel_pruning:
                mask = torch.ones(net_weight.size(self.pruning_dim)).to(net_weight.device)
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


class SoftPruningParametrization(nn.Module):
    # Parametrization technique where the weights are not completely zeroed out, however, they are pruned to zero incrementally with every epoch
    def __init__(self, curr_node, modules, channel_pruning=False, pruning_ratio=0.6, n2m_pruning=False,  
                 init_train_ep=5, net_weight = None, binary_mask=False, tao=1e-4, pruning_dim = 0, **kwargs):
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
        is_depthwise = (self.curr_node.target in self.all_modules) and isinstance(self.all_modules[self.curr_node.target],nn.Conv2d) and (self.all_modules[self.curr_node.target].weight.shape[1] == 1) # we do not want to prune depthwise layers
        if int(self.pruning_ratio*net_weight.nelement())==0 or is_depthwise: # do not prune layers with less than 32 channels only for channel pruning
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
    def __init__(self, curr_node, modules, channel_pruning=False, pruning_ratio=0.6, n2m_pruning=False, init_train_ep=5, net_weight=None, binary_mask=False, tao=0.0001,p=2,pruning_dim =0, **kwargs):
        self.p =p
        super().__init__(curr_node, modules, channel_pruning, pruning_ratio, n2m_pruning, init_train_ep, net_weight, binary_mask, tao, pruning_dim=pruning_dim, **kwargs)
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


#Experimental to check either only Channel Pruning works for  MultiHeadAttention Layer's inner_projection weights
class ChannelOnlyBlendPruningParametrization(BlendPruningParametrization):
    def __init__(self, curr_node, modules, channel_pruning=False, pruning_ratio=0.6, n2m_pruning=False, init_train_ep=5, net_weight=None, binary_mask=False, tao=0.0001, p=2,pruning_dim = 0, **kwargs):
        self.old_shape = list(net_weight.shape)
        assert len(self.old_shape) == 2, 'This parameterization is only for \'MultiHeadAttention\' Layer\'s in_proj weight and bias. \n So, weight of the node given must have dimension of 2'
        assert  self.old_shape[0]%self.old_shape[1] == 0, 'The number of heads should be obtainable from weights i.e., weight.shape[0] must be divisible by weight.shape[1] '
        if curr_node.op == 'call_module':
            module = modules[curr_node.target]
            if isinstance(module,(nn.MultiheadAttention,)):
                self.num_heads = module.num_heads 
            elif isinstance(module,(nn.Linear)):
                module_name,attr = curr_node.target.rsplit('.',1)
                module = modules[module_name]
                if has_timm and isinstance(module,(tmmodels.vision_transformer.Attention,tmmodels.swin_transformer.WindowAttention)):
                    self.num_heads = module.num_heads
                else:
                    raise Exception('This parametrization is only for inner projection layer of attention layers (for timm)')
        elif curr_node.op == 'call_function':
            if has_tv and curr_node.target == tvmodels.swin_transformer.shifted_window_attention:
                self.num_heads = list(curr_node.args)[-1]
            else:
                raise Exception('This parametrization is only for inner projection layer of shifted window attention layers (for torchvision)')
        else:
            raise Exception('This Parameterization is only for different \'Attention\' Layer\'s in_proj weight and bias')
        self.shape1 =[3,self.num_heads,self.old_shape[0]//(3*self.num_heads),self.old_shape[1]]
        super().__init__(curr_node, modules, channel_pruning, pruning_ratio, n2m_pruning, init_train_ep, net_weight, binary_mask, tao, p,pruning_dim, **kwargs)
        
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
    def __init__(self, curr_node, modules, channel_pruning=False, pruning_ratio=0.6, n2m_pruning=False, init_train_ep=5, net_weight=None, binary_mask=False, tao=0.0001, p=2, pruning_dim = 0, **kwargs):
        self.old_shape = list(net_weight.shape)
        assert len(self.old_shape) == 2, 'This parameterization is only for \'MultiHeadAttention\' Layer\'s in_proj weight and bias. \n So, weight of the node given must have dimension of 2'
        assert  self.old_shape[0]%self.old_shape[1] == 0, 'The number of heads should be obtainable from weights i.e., weight.shape[0] must be divisible by weight.shape[1] '
        if curr_node.op == 'call_module':
            module = modules[curr_node.target]
            if isinstance(module,(nn.MultiheadAttention,)):
                self.num_heads = module.num_heads 
            elif isinstance(module,(nn.Linear)):
                module_name,attr = curr_node.target.rsplit('.',1)
                module = modules[module_name]
                if has_timm and isinstance(module,(tmmodels.vision_transformer.Attention,tmmodels.swin_transformer.WindowAttention)):
                    self.num_heads = module.num_heads
                else:
                    raise Exception('This parametrization is only for inner projection layer of attention layers (for timm)')
        elif curr_node.op == 'call_function':
            if has_tv and curr_node.target == tvmodels.swin_transformer.shifted_window_attention:
                self.num_heads = list(curr_node.args)[-1]
            else:
                raise Exception('This parametrization is only for inner projection layer of shifted window attention layers (for torchvision)')
        else:
            raise Exception('This Parameterization is only for different \'Attention\' Layer\'s in_proj weight and bias')
        self.shape1 =[3,self.num_heads,self.old_shape[0]//(3*self.num_heads),self.old_shape[1]]
        super().__init__(curr_node, modules, channel_pruning, pruning_ratio, n2m_pruning, init_train_ep, net_weight, binary_mask, tao, p,pruning_dim, **kwargs)
        
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
    def __init__(self, curr_node, modules, channel_pruning=False, pruning_ratio=0.6, n2m_pruning=False, init_train_ep=5, net_weight=None, binary_mask=False, tao=0.0001, p=2, pruning_dim = 0, **kwargs):
        self.old_shape = list(net_weight.shape)
        assert len(self.old_shape) == 2, 'This parameterization is only for \'MultiHeadAttention\' Layer\'s in_proj weight and bias. \n So, weight of the node given must have dimension of 2'
        assert  self.old_shape[0]%self.old_shape[1] == 0, 'The number of heads should be obtainable from weights i.e., weight.shape[0] must be divisible by weight.shape[1] '
        if curr_node.op == 'call_module':
            module = modules[curr_node.target]
            if isinstance(module,(nn.MultiheadAttention,)):
                self.num_heads = module.num_heads 
            elif isinstance(module,(nn.Linear)):
                module_name,attr = curr_node.target.rsplit('.',1)
                module = modules[module_name]
                if has_timm and isinstance(module,(tmmodels.vision_transformer.Attention,tmmodels.swin_transformer.WindowAttention)):
                    self.num_heads = module.num_heads
                else:
                    raise Exception('This parametrization is only for inner projection layer of attention layers (for timm)')
        elif curr_node.op == 'call_function':
            if has_tv and curr_node.target == tvmodels.swin_transformer.shifted_window_attention:
                self.num_heads = list(curr_node.args)[-1]
            else:
                raise Exception('This parametrization is only for inner projection layer of shifted window attention layers (for torchvision)')
        else:
            raise Exception('This Parameterization is only for different \'Attention\' Layer\'s in_proj weight and bias')
        self.shape1 =[3,self.num_heads,self.old_shape[0]//(3*self.num_heads),self.old_shape[1]]
        super().__init__(curr_node, modules, channel_pruning, pruning_ratio, n2m_pruning, init_train_ep, net_weight, binary_mask, tao, p,pruning_dim, **kwargs)
        
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


PRUNING_CLASS_DICT = {"blend": BlendPruningParametrization, 
                 "sigmoid": SigmoidPruningParametrization, 
                 "incremental": IncrementalPruningParametrization}