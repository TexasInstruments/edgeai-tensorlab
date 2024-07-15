import torch
from torch import fx
from typing import Callable, Any,Union, Dict, Optional, List#, override
from torch.nn.modules import Module
from .custom_modules import ReplaceBatchNorm



class CustomTracer(fx.Tracer):
    def __init__(self,*args,**kwargs ) -> None:
        super().__init__(*args,**kwargs)
        self.custom_leaf_module_types = []
    
    # @override()
    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool:
        return super().is_leaf_module(m, module_qualified_name) or any(isinstance(m,t) for t in self.custom_leaf_module_types)
    
    def add_custom_leaf_module_type(self,m_type):
        self.custom_leaf_module_types.append(m_type)
    
    def add_custom_leaf_module_types(self,modules:List):
        self.custom_leaf_module_types.extend(modules)


default_tracer = CustomTracer()
default_tracer.add_custom_leaf_module_types([
    ReplaceBatchNorm,
])

class AddModule(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self,x,y):
        return x+y


# default_tracer.add_custom_leaf_module_type(AddModule)


def custom_symbolic_trace( root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[Dict[str, Any]] = None,
    custom_tracer:fx.Tracer = None 
    ):
    tracer = custom_tracer or default_tracer
    graph = tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    return fx.GraphModule(tracer.root, graph, name)


if __name__ =='__main__':
    class ExampleModule(Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.double_layer  = AddModule()
        
        def forward(self,x):
            return self.double_layer(x,x)

    model = ExampleModule()

    fx_model1 = custom_symbolic_trace(model)
    fx_model2 = fx.symbolic_trace(model)
    print(fx_model1.graph)
    print(fx_model2.graph)
    fx_model1.print_readable()
    fx_model2.print_readable()
    
