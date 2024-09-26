from .optimization_utils import *
from ..surgery.v1 import SurgeryModule 
from ..quantization.v1 import QuantTrainModule  

class ModelOptimizationWrapperV1(SurgeryModule, QuantTrainModule):
    def __init__(self, model, *args, replacement_dict=None, transformation_dict=None, copy_attrs=None, **kwargs) -> None:
        copy_attrs= copy_attrs or []
        super().__init__(model, *args, replacement_dict=replacement_dict, transformation_dict=transformation_dict, copy_attrs=copy_attrs, **kwargs)