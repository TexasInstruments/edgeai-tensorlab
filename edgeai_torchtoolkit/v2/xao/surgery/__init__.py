import enum

from . import custom_modules, custom_surgery_functions,surgery
from .surgery import SurgeryModule, replace_unsuppoted_layers, get_replacement_dict_default

class ModelSyrgeryType(enum.Enum):
    NO_SURGERY = 0
    MODEL_SURGERY_LEGACY = 1
    MODEL_SURGERY_FX = 2

    def __str__(self):
        return self.name
    
    def get_dict():
        return {t.value:t.name for t in __class__}

