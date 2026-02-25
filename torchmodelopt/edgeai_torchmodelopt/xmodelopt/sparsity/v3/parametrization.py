import torch
from torch import nn, fx, Tensor
from torch.fx.passes.utils.source_matcher_utils import  SourcePartition
from torch.fx.passes.utils.matcher_utils import InternalMatch
import math
import warnings

from .... import xnn

SPARSITY_CLASS_DICT = {}

def register_class(name, cls=None):
    """Registers a sparsity parametrization class with the global SPARSITY_CLASS_DICT.
    
    This decorator function allows sparsity classes to be registered by name, making them
    available for use in the sparsity system. When a class is registered, it also checks
    for the REQUIRED_SPARSE_PARAMS attribute and warns if it's missing.
    
    Args:
        name (str): The name to register the class under (e.g., 'n2m' for N2MSparsityParametrization).
        cls (class, optional): The class to register. If None, returns a decorator.
        
    Returns:
        If cls is provided, returns the registered class.
        If cls is None, returns a decorator function that will register the decorated class.
        
    Example:
        @register_class('n2m')
        class N2MSparsityParametrization(BaseSparsityParametrization):
            # Class implementation
    """
    def _registered(cls):
        if not hasattr(cls, 'REQUIRED_SPARSE_PARAMS'):
            warnings.warn(f'class {cls.__name__} does not have REQUIRED_SPARSE_PARAMS which is required to collect params from module.__sparse_param__')
            cls.REQUIRED_SPARSE_PARAMS = []
        SPARSITY_CLASS_DICT[name]=cls
        return cls
    if cls is not None:
        return _registered(name)
    return _registered



class BaseSparsityParametrization(nn.Module):
    """Base class for all sparsity parametrization implementations.
    
    This class provides the foundation for sparsity parametrization, defining the core
    functionality for creating and applying sparsity masks to weight tensors. It uses
    the PyTorch parametrization system to modify the behavior of weights during forward passes.
    
    """
    # REQUIRED_SPARSE_PARAMS = ['epoch_count', 'init_train_ep', 'total_epochs', 'p',] 

    class_mask_func_dict = {}
    class_forward_func_dict = {}
    
    def __init__(self, source, nodes, *args, tensor=None, current_epoch=0, sparsity_start_epoch=5, 
                 sparsity_end_epoch=15, binary_mask=False, p=None, **kwargs):
        """Initializes the base sparsity parametrization.
        
        Args:
            source: Source identifier for the parametrization (e.g., layer type tuple).
            nodes: Graph nodes to which the parametrization applies.
            *args: Additional arguments passed to the parent class.
            tensor: The weight tensor to apply sparsity to. Required.
            current_epoch (int, optional): Current training epoch. Defaults to 0.
            init_train_ep (int, optional): Initial training epochs before sparsification. Defaults to 5.
            total_epochs (int, optional): Total number of training epochs. Defaults to 15.
            binary_mask (bool, optional): Whether to use binary (hard) masks. Defaults to False.
            p (float, optional): Power parameter for sparsity schedule. Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent class.
            
        Note:
            The parametrization creates masks for sparsity, which are either soft masks
            during training (gradually increasing sparsity) or hard binary masks for
            final evaluation/inference.
        """
        # super().__init__(*args, **kwargs)
        self.source = source
        self.nodes = nodes
        self.current_epoch = current_epoch
        self.sparsity_start_epoch = sparsity_start_epoch
        self.sparsity_end_epoch = sparsity_end_epoch
        self.mask_func_dict = {}
        self.forward_func_dict = {}
        self.binary_mask = binary_mask
        self.p = p
        self.alpha_factor = None # used for debugging alpha value

        super().__init__(*args, **kwargs)
        
        # Register mask and forward functions based on the derived class implementation
        self.register_masks()
        self.register_forwards()
        
        # Create the sparsity mask from the tensor
        tensor = tensor.detach()
        self.mask = self.create_mask(tensor)
        
    def register_mask(self, *args, func=None):
        """Registers a mask creation function for this parametrization instance.
        
        This method allows registering custom mask generation functions for specific
        sparsity patterns and layer types. The registered functions are stored in the
        global class_mask_func_dict with a key that includes the current instance and args.
        
        Args:
            *args: Variable length arguments used as part of the key for the mask function.
                  Typically includes layer type and sparsity pattern information.
            func: The mask function to register. If None, returns a decorator.
            
        Returns:
            If func is provided, returns the registered function.
            If func is None, returns a decorator function that will register the decorated function.
            
        Example:
            @self.register_mask('Conv2d', 2, 4, 'n2m')
            def conv_mask_gen(tensor):
                # Mask generation logic
                return mask
        """
        def _registered(func):
            self.class_mask_func_dict[(self,args)] = func
            return func
        if func is None:
            return _registered
        return _registered(func)
    
    def register_forward(self, *args, func=None):
        """Registers a forward function for this parametrization instance.
        
        This method allows registering custom forward pass functions for specific
        sparsity patterns and layer types. The registered functions are stored in the
        global class_forward_func_dict with a key that includes the current instance and args.
        
        Args:
            *args: Variable length arguments used as part of the key for the forward function.
                  Typically includes layer type and sparsity pattern information.
            func: The forward function to register. If None, returns a decorator.
            
        Returns:
            If func is provided, returns the registered function.
            If func is None, returns a decorator function that will register the decorated function.
            
        Example:
            @self.register_forward('Conv2d', 2, 4, 'n2m')
            def conv_forward(X):
                # Forward pass implementation
                return X * self.mask
        """
        def _registered(func):
            self.class_forward_func_dict[(self,args)] = func
            return func
        if func is None:
            return _registered
        return _registered(func)
    
    def register_masks(self):
        """Registers mask generation functions for this parametrization.
        
        This method should be implemented by derived classes to register specific
        mask generation functions for different layer types and sparsity patterns.
        Each derived class should use self.register_mask() to register appropriate
        mask generation functions.
        
        Raises:
            NotImplementedError: If not implemented by the derived class.
        """
        raise NotImplementedError
    
    def register_forwards(self):
        """Registers forward pass functions for this parametrization.
        
        This method should be implemented by derived classes to register specific
        forward pass functions for different layer types and sparsity patterns.
        Each derived class should use self.register_forward() to register appropriate
        forward functions.
        
        Raises:
            NotImplementedError: If not implemented by the derived class.
        """
        raise NotImplementedError
    
    def get_mask_func(self, name):
        """Retrieves the mask generation function for the specified name.
        
        Args:
            name: The name/key (usually a tuple) identifying the mask function.
            
        Returns:
            function: The registered mask generation function.
            
        Raises:
            KeyError: If no mask function is registered with the given name.
        """
        return self.class_mask_func_dict[(self, name)]
    
    def get_forward_func(self, name, default=None):
        """Retrieves the forward function for the specified name.
        
        Args:
            name: The name/key (usually a tuple) identifying the forward function.
            default (function, optional): Default function to return if no function
                is registered with the given name. Defaults to None.
                
        Returns:
            function: The registered forward function or the default function if provided.
            
        Raises:
            KeyError: If no forward function is registered with the given name and
                no default is provided.
        """
        if default:
            return self.class_forward_func_dict.get((self, name), default)
        return self.class_forward_func_dict[(self, name)]
    
    def create_mask(self, tensor):
        """Creates a sparsity mask for the given tensor.
        
        This method calls the appropriate mask generation function based on the source
        identifier, which typically includes layer type and sparsity pattern information.
        
        Args:
            tensor (torch.Tensor): The weight tensor to create a mask for.
            
        Returns:
            torch.Tensor: A mask tensor of the same shape as the input tensor.
                Values are typically between 0 and 1, with 0 representing elements
                to be pruned and 1 representing elements to keep.
        """
        mask = self.get_mask_func(self.source)(tensor)
        # Apply binary thresholding if requested, otherwise use the soft mask
        self.mask = (mask >= 0.5) if self.binary_mask else mask
        return self.mask

    def update(self, tensor=None, mask=None, binary_mask=False):
        """Update self, assuming an epoch has been completed. Update state (self.current_epoch) to reflect this.
        If mask != None, set the input mask as current mask
        If tensor != None, create a mask from tensor, based on the new self.current_epoch

        Args:
            tensor (_type_, optional): Input tensor to create mask for. Defaults to None.
            mask (_type_, optional): Input mask to directly set. Defaults to None.
            binary_mask (bool, optional): If True, use hard mask (see self.create_mask). Defaults to False.

        Raises:
            ValueError
        
        Returns: 
            new updated mask, also sets self.mask
        """
        if mask is not None:
            self.mask = mask 
            return self.mask  
        self.current_epoch += 1
        self.binary_mask = binary_mask
        if tensor is None:
            raise ValueError(f"update_mask got tensor={tensor} with no mask input")
        if self.freeze_mask and self.current_epoch > self.sparsity_end_epoch:
            # Do not change mask
            return self.mask
        self.mask = self.create_mask(tensor)
        return self.mask 
        
    
    def forward(self, X):
        """Applies the sparsity mask during the forward pass.
        
        This method is called by PyTorch's parametrization system during forward passes
        through the network. It applies the appropriate forward function based on the
        source identifier, with a default implementation of element-wise multiplication
        with the mask.
        
        Args:
            X (torch.Tensor): The original weight tensor.
            
        Returns:
            torch.Tensor: The modified weight tensor with sparsity applied.
        """
        default = lambda x: x * self.mask
        return self.get_forward_func(self.source, default)(X)
        
    def get_alpha_factor(self):
        """Calculates the alpha factor for gradual sparsification.
        
        This method computes a factor that controls the sparsity level during training,
        implementing a gradual transition from no sparsity to full sparsity. The
        schedule has three phases:
        1. Initial phase: Full weights (alpha=1)
        2. Transition phase: Gradually decreasing alpha based on a power function
        3. Final phase: Full sparsity (alpha=0)
        
        Returns:
            float: Alpha factor between 0 and 1, where:
                - 1 means no sparsity (keep all weights)
                - 0 means full sparsity (apply the full mask)
                - Values between 0 and 1 create soft masks during training
                
        Note:
            The transition follows a power function curve controlled by self.p,
            which determines how quickly sparsity increases during training.
        """
        # Calculate the knee point (where transition phase ends and final phase begins)
        # total_epochs_knee_point = min(
        #     self.total_epochs - 1,
        #     max(self.init_train_ep + 1, (self.total_epochs - self.init_train_ep) * 2 // 3 + self.init_train_ep)
        # )

        # Insted of knee point, we will use sparsity_start_epoch sparsity_end_epoch
        # total_epochs_knee_point = self.incremental_epochs-1
        
        # # Initial phase: No sparsity
        # if self.epoch_count <= self.init_train_ep:
        #     alpha_factor = 1
        # # Final phase: Full sparsity
        # elif self.epoch_count > total_epochs_knee_point:
        #     alpha_factor = 0
        # # Transition phase: Gradually increase sparsity
        # else:
        #     alpha_factor = math.pow(
        #         abs(total_epochs_knee_point - self.epoch_count), self.p
        #     ) / math.pow(
        #         total_epochs_knee_point - self.init_train_ep, self.p
        #     )
        # self.alpha_factor = alpha_factor
        
        if self.current_epoch <= self.sparsity_start_epoch:
            # Initial phase: No sparsity
            alpha_factor = 1
        elif self.current_epoch > self.sparsity_end_epoch:
            # Final phase: Full sparsity
            alpha_factor = 0
        else:
            # Transition phase: Gradually increase sparsity, alpha from 1 to 0
            alpha_factor = math.pow(
                abs(self.sparsity_end_epoch - self.current_epoch), self.p
            ) / math.pow(
                self.sparsity_end_epoch - self.sparsity_start_epoch, self.p
            )

        self.alpha_factor = alpha_factor
        return self.alpha_factor

STE_GAMMA = 0

class MaskMul(torch.autograd.Function):
    @staticmethod
    def forward(weights, mask):
        output = weights * mask
        return output
    
    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of forward().
    def setup_context(ctx, inputs, output):
        weights, mask = inputs
        ctx.save_for_backward(weights, mask)
    
    @staticmethod
    def backward(ctx, grad_output):
        weights, mask = ctx.saved_tensors
        mask_inverse = (1-mask)
        
        global STE_GAMMA
        total_grad = grad_output * mask + STE_GAMMA*weights*(1-mask)
        return total_grad, None

@register_class('n2m')
class N2MSparsityParametrization(BaseSparsityParametrization):
    """Implementation of n:m structured sparsity parametrization.
    
    This class implements the n:m structured sparsity pattern, where in each block of
    m elements, only n elements are kept (non-zero) and the rest are pruned (set to zero).
    This results in a sparsity ratio of (m-n)/m. For example, 2:4 sparsity keeps 2 out of
    every 4 elements, resulting in 50% sparsity.
    
    Attributes:
        REQUIRED_SPARSE_PARAMS (list): List of parameter names required from module.__sparse_params__
            when applying the parametrization, extending the base class list with 'n' and 'm'.
    """
    # REQUIRED_SPARSE_PARAMS = ['n', 'm', 'mode', 'incremental_epochs'] + BaseSparsityParametrization.REQUIRED_SPARSE_PARAMS
    
    # def __init__(self, source, nodes, *args, n=None, m=None, tensor=None, epoch_count=0, 
    #              init_train_ep=5, total_epochs=15, binary_mask=False, p=None, mode=None, **kwargs):
    def __init__(self, source, nodes, *args, n:int =None, m:int =None, mode:str='topk', p:int=2, tensor:Tensor, freeze_mask:bool=False, **kwargs):
        """Initializes the n:m sparsity parametrization.

        Args:
            source (_type_): Source identifier for the parametrization (e.g., layer type tuple).
            nodes (_type_): Graph nodes to which the parametrization applies.
            n (int): The n value in n:m pattern (number of non-zero elements per block). Required.
            m (int): The m value in n:m pattern (block size). Required.
            mode (str, optional): Method for selecting elements to keep. Options are:
                - 'topk': Keep top-k elements by magnitude (default)
                - 'topk_blockwise': Keep top-k blocks by magnitude, but perform incremental sparsity blockwise;
                    i.e. x% of blocks have n weights set to zero at each epoch
                - 'magnitude': Alternative magnitude-based selection (not implemented)
                - 'hessian': Hessian-based selection (not implemented)
            tensor (torch.Tensor): The weight tensor to create a mask for.
            p (int): Power law scaling parameter for incremental sparsity
            freeze_mask (bool): If true, freeze mask after sparsity_end_epoch
            For other kwargs, see BaseSparsityParametrization
        Raises:
            AssertionError: If n, m, or tensor is not provided, or if mode is invalid.
        """
        assert n is not None and m is not None and tensor is not None, f'n, m and tensor has to be provided'
        self.n = n
        self.m = m
        self.mode = mode
        self.freeze_mask = freeze_mask
        assert self.mode in ('topk', 'topk_blockwise', 'magnitude', 'hessian')
        
        # Add mode to the source identifier to differentiate between different selection methods
        source += (self.mode,)
        
        super().__init__(source, nodes, *args, tensor=tensor, p=p, **kwargs)
    
    def get_topk_mask(self, tensor, alpha_factor):
        """Creates a sparsity mask using the top-k method.
        
        This method creates a mask that keeps the top-n elements (by magnitude) in each
        block of m elements, implementing the n:m sparsity pattern. Elements not in the
        top-n are assigned the alpha_factor value, which controls the degree of sparsity
        during training.
        
        Args:
            tensor (torch.Tensor): The weight tensor to create a mask for.
            alpha_factor (float): The alpha factor (between 0 and 1) for soft masks.
                Values closer to 0 increase sparsity, while values closer to 1 reduce it.
                
        Returns:
            torch.Tensor: A mask tensor with the same shape as the input tensor.
                Elements to keep have value 1, elements to prune have value alpha_factor.
                
        Note:
            The method works by:
            1. Taking the absolute value of the tensor
            2. Reshaping it to group elements into blocks of size m
            3. Finding the bottom n elements (by magnitude) in each block
            4. Setting those elements to alpha_factor in the mask
            5. Reshaping the mask back to the original tensor shape
        """
        # Take absolute value to consider magnitude only
        tensor = torch.abs(tensor)
        shape = tensor.shape
        
        # Reshape to group elements into blocks of size m
        tensor_reshaped = tensor.view(-1, self.m)
        
        # Initialize mask with all ones (keep all elements)
        soft_mask = torch.ones_like(tensor_reshaped)
        
        # Find the n smallest elements in each block
        # Note: largest=False means we're finding the smallest values
        wl = torch.topk(tensor_reshaped, self.n, -1, largest=False, sorted=True)
        
        if self.mode == 'topk':
            # Set those smallest elements to alpha_factor in the mask
            # This effectively reduces their contribution during training
            soft_mask.scatter_(-1, wl.indices, alpha_factor)
        elif self.mode == 'topk_blockwise':
            # is there a more efficient method? TODO 
            # Since topk output is sorted, following gives us the max among pruned weights (i.e. largest among n smallest)
            max_chosen = wl.indices[:,1]
            max_chosen_values = tensor_reshaped.gather(1, max_chosen.unsqueeze(-1)).squeeze(-1)
            # Get the (1-alpha) smallest fraction so the largest pruned weight is minimized
            pruned = torch.topk(max_chosen_values, int((1.0-alpha_factor)*len(max_chosen_values)), largest=False)
            # For each 'row' in the alpha fraction above, prune the n smallest identified by wl
            soft_mask[pruned.indices] = soft_mask[pruned.indices].scatter(1, wl.indices[pruned.indices], 0) 
        else:
            raise NotImplementedError
        
        # Reshape back to original tensor shape
        return soft_mask.view(shape)
    
    def get_magnitude_mask(self, tensor, alpha_factor):
        """Creates a sparsity mask using the magnitude method.
        
        This method would create a mask based on element magnitudes using an alternative
        approach to the top-k method. However, it is not currently implemented.
        
        Args:
            tensor (torch.Tensor): The weight tensor to create a mask for.
            alpha_factor (float): The alpha factor for soft masks.
                
        Returns:
            torch.Tensor: A mask tensor.
                
        Raises:
            NotImplementedError: This method is not yet implemented.
            
        TODO: Implement magnitude-based mask generation as an alternative to top-k.
        """
        raise NotImplementedError
    
    def get_hessian_mask(self, tensor, alpha_factor):
        """Creates a sparsity mask using the Hessian method.
        
        This method would create a mask based on the Hessian (second derivatives),
        which can provide information about parameter importance. However, it is not
        currently implemented.
        
        Args:
            tensor (torch.Tensor): The weight tensor to create a mask for.
            alpha_factor (float): The alpha factor for soft masks.
                
        Returns:
            torch.Tensor: A mask tensor.
                
        Raises:
            NotImplementedError: This method is not yet implemented.
            
        TODO: Implement Hessian-based mask generation for more advanced pruning.
        """
        raise NotImplementedError
    
    def register_masks(self):
        """Registers mask generation functions for n:m sparsity.
        
        This method sets up mask generation functions for Conv2d, Linear, and matmul
        operations with n:m sparsity. All layer types use a common basic mask generation
        function that:
        1. Returns all ones during initial training epochs (no sparsity)
        2. Uses the specified mode (topk, magnitude, hessian) to generate masks during
           the sparsification phase
        
        Note:
            Currently only the topk method is fully implemented. The magnitude and hessian
            methods are placeholders for future implementation.
        """
        # Map mode names to their corresponding mask generation methods
        mode_2_func_dict = dict(
            topk = self.get_topk_mask,
            topk_blockwise = self.get_topk_mask,
            magnitude = self.get_magnitude_mask,
            hessian = self.get_hessian_mask,
        )
        
        def basic_mask_gen(tensor):
            """Basic mask generation function used by all layer types.
            
            During initial training epochs, returns all ones (no sparsity).
            After initial training, uses the selected mode to generate masks
            with appropriate alpha factor.
            
            Args:
                tensor (torch.Tensor): The weight tensor to create a mask for.
                
            Returns:
                torch.Tensor: A mask tensor.
            """
            # NOTE: this should not be necessary, since it's handled by get_alpha_factor.
            # It would probably be better to allow mask_gen_func to handle alpha computation as well
            if self.current_epoch <= self.sparsity_start_epoch:
                # No sparsity during initial training epochs
                return torch.ones_like(tensor)
            else:
                # Calculate alpha factor and generate mask using selected mode
                alpha_factor = self.get_alpha_factor()
                return mode_2_func_dict[self.mode](tensor, alpha_factor)
        
        # Register mask generators for different layer types
        # Note: All currently use the same basic_mask_gen function
        
        @self.register_mask('Conv2d', self.n, self.m, 'n2m', self.mode)
        def conv_mask_gen(tensor):
            """Generates masks for Conv2d layers with n:m sparsity.
            
            Args:
                tensor (torch.Tensor): The weight tensor to create a mask for.
                
            Returns:
                torch.Tensor: A mask tensor.
            """
            # Commented code shows how specialized handling could be added
            # if self.mode == 'hessian':
            #     pass
            return basic_mask_gen(tensor)
        
        @self.register_mask('Linear', self.n, self.m, 'n2m', self.mode)
        def linear_mask_gen(tensor):
            """Generates masks for Linear layers with n:m sparsity.
            
            Args:
                tensor (torch.Tensor): The weight tensor to create a mask for.
                
            Returns:
                torch.Tensor: A mask tensor.
            """
            # if self.mode == 'hessian':
            #     pass
            return basic_mask_gen(tensor)
        
        @self.register_mask('matmul', self.n, self.m, 'n2m', self.mode)
        def matmul_mask_gen(tensor):
            """Generates masks for matmul operations with n:m sparsity.
            
            Args:
                tensor (torch.Tensor): The weight tensor to create a mask for.
                
            Returns:
                torch.Tensor: A mask tensor.
            """
            # if self.mode == 'hessian':
            #     pass
            return basic_mask_gen(tensor)
    
    def register_forwards(self):
        """Registers forward functions for n:m sparsity.
        
        This method sets up forward pass functions for Conv2d, Linear, and matmul
        operations with n:m sparsity. All layer types use a common default forward
        function that applies the sparsity mask to the weight tensor via element-wise
        multiplication.
        """
        def default_forward(X):
            """Default forward function used by all layer types.
            
            Simply applies the sparsity mask to the input tensor via
            element-wise multiplication.
            
            Args:
                X (torch.Tensor): The input weight tensor.
                
            Returns:
                torch.Tensor: The masked weight tensor.
            """
            # return self.mask * X
            return MaskMul.apply(X, self.mask)
        
        # Register forward functions for different layer types
        # Note: All currently use the same default_forward function
        
        @self.register_forward('Conv2d', self.n, self.m, 'n2m', self.mode)
        def conv_forward(X):
            return default_forward(X)
        
        @self.register_forward('Linear', self.n, self.m, 'n2m', self.mode)
        def linear_forward(X):
            return default_forward(X)
        
        @self.register_forward('matmul', self.n, self.m, 'n2m', self.mode)
        def matmul_forward(X):
            return default_forward(X)        
            
