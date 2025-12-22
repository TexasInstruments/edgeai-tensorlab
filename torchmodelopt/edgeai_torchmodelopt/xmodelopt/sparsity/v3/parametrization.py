import torch
from torch import nn, fx
from torch.fx.passes.utils.source_matcher_utils import  SourcePartition
from torch.fx.passes.utils.matcher_utils import InternalMatch
import math

from .... import xnn

SPARSITY_CLASS_DICT = {}

def register_class(name, cls=None):
    def _registered(cls):
        SPARSITY_CLASS_DICT[name]=cls
        return cls
    if cls is not None:
        return _registered(name)
    return _registered

class_mask_func_dict = {}
class_forward_func_dict = {}

class BaseSparsityParametrization(nn.Module):
    def __init__(self, source, nodes, *args, tensor=None, epoch_count=0, init_train_ep=5, total_epochs=15, binary_mask=False, p=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = source
        self.nodes= nodes
        self.epoch_count = epoch_count
        self.init_train_ep = init_train_ep
        self.total_epochs = total_epochs
        self.mask_func_dict =  {}
        self.forward_func_dict =  {}
        self.binary_mask = binary_mask
        self.p = p
        self.register_masks()
        self.register_forwards()
        tensor = tensor.detach()
        mask = self.create_mask(tensor)
        self.mask = (mask>=0.5) if self.binary_mask  else mask
        
    def register_mask(self, *args, func=None):
        def _registered(func):
            class_mask_func_dict[(self,args)] = func
            return func
        if func is None:
            return _registered
        return _registered(func)
    
    def register_forward(self, *args, func=None):
        def _registered(func):
            class_forward_func_dict[(self,args)] = func
            return func
        if func is None:
            return _registered
        return _registered(func)
    
    def register_masks(self):
        raise NotImplementedError
    
    def register_forwards(self):
        raise NotImplementedError
    
    def get_mask_func(self, name):
        return class_mask_func_dict[(self, name)]
    
    def get_forward_func(self, name, default=None):
        if default:
            return class_forward_func_dict.get((self, name), default)
        return class_forward_func_dict[(self, name)]
    
    def create_mask(self, tensor):
        return self.get_mask_func(self.source)(tensor)
    
    def forward(self, X):
        default = lambda x : x*self.mask
        return self.get_forward_func(self.source, default)(X)
        
    def get_alpha_factor(self,):
        total_epochs_knee_point = min(self.total_epochs-1, max(self.init_train_ep+1, (self.total_epochs-self.init_train_ep)*2//3 + self.init_train_ep))
        if self.epoch_count<=self.init_train_ep:
            alpha_factor = 1
        elif self.epoch_count>total_epochs_knee_point:
            alpha_factor = 0
        else:
            alpha_factor = math.pow(abs(total_epochs_knee_point - self.epoch_count),self.p)/math.pow(total_epochs_knee_point-self.init_train_ep, self.p)
        return alpha_factor


@register_class('n2m')
class N2MSparsityParametrization(BaseSparsityParametrization):
    def __init__(self, source, nodes, *args, n=None, m=None, tensor=None, epoch_count=0, init_train_ep=5, total_epochs=15, binary_mask=False, p=None, mode=None,**kwargs):
        assert n is not None and m is not None and tensor is not None, f'n, m and tensor has to be provided'
        self.n = n
        self.m = m
        self.init_train_ep = init_train_ep
        self.binary_mask = binary_mask
        self.p  = p or 2
        self.mode = mode or 'topk' # or hessian or magnitude
        assert self.mode in ('topk', 'magnitude','hessian')
        source += (self.mode,)
        super().__init__(source,nodes, tensor=tensor, epoch_count=epoch_count, init_train_ep=init_train_ep, total_epochs=total_epochs, binary_mask=binary_mask,p=p)
    
    def get_topk_mask(self, tensor, alpha_factor):
        tensor = torch.abs(tensor)
        shape = tensor.shape
        tensor_reshaped = tensor.view(-1, self.m)
        soft_mask = torch.ones_like(tensor_reshaped)
        wl = torch.topk(tensor_reshaped, self.n, -1, largest=False)
        soft_mask.scatter_(-1, wl.indices, alpha_factor)
        return soft_mask.view(shape)
    
    def get_magnitude_mask(self, tensor, alpha_factor):
        raise NotImplementedError
    
    def get_hessian_mask(self, tensor, alpha_factor):
        raise NotImplementedError
    
    def register_masks(self):
        
        mode_2_func_dict = dict(
            topk = self.get_topk_mask,
            magnitude=self.get_magnitude_mask,
            hessian=self.get_hessian_mask,
        )
        
        def basic_mask_gen(tensor):
            if self.epoch_count<=self.init_train_ep:
                return torch.ones_like(tensor)
            else:
                alpha_factor = self.get_alpha_factor()
                return mode_2_func_dict[self.mode](tensor, alpha_factor)
        
        @self.register_mask('Conv2d', self.n, self.m, 'n2m', self.mode)
        def conv_mask_gen(tensor):
            # if self.mode == 'hessian':
            #     pass
            return basic_mask_gen(tensor)
        
        @self.register_mask('Linear', self.n, self.m, 'n2m', self.mode)
        def conv_mask_gen(tensor):
            # if self.mode == 'hessian':
            #     pass
            return basic_mask_gen(tensor)
    
    def register_forwards(self):
        def default_forward(X):
            return self.mask*X
        
        @self.register_forward('Conv2d', self.n, self.m, 'n2m', self.mode)
        def conv_forward(X):
            return default_forward(X)
        
        @self.register_forward('Linear', self.n, self.m, 'n2m', self.mode)
        def linear_forward(X):
            return default_forward(X)
        
        
            
