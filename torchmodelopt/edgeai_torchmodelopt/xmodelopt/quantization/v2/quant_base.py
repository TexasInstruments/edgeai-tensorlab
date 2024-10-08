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

import types
import torch
from . import quant_func_wrapper
from ...utils.optimization_base import OptimizationBaseModule, add_attrs


class QuantFxBaseModule(OptimizationBaseModule):
    def __init__(self, model, *args, transformation_dict:dict=None, copy_attrs:list[str]=None, add_methods=True, **kwargs):
        '''
        model: input model to be used for QAT / PTC
        qconfig_type: qconfig_type can be one of the modes defined in qconfig_types (string)
            or it can be a dict that will be passed to qconfig_types.get_config_from_dict()
            it can also be an instance of torch.ao.quantization.QConfig as used when using torch.ao.quantization apis
        transformation_dict: 
        '''
        # self.module = quant_func.init(model, *args, add_methods=add_methods, **kwargs)
        copy_attrs= copy_attrs or []
        super().__init__(model, *args, transformation_dict=transformation_dict, copy_attrs=copy_attrs, **kwargs)
        self.prepare(self.module, *args, transformation_dict=self.transformation_dict, add_methods=add_methods,**kwargs)
        
    def prepare(self, model, *args, transformation_dict=None, add_methods=True, **kwargs):
        assert isinstance(self, OptimizationBaseModule), 'This only works if self is OptimizationBaseModule object'
        self.module =quant_func_wrapper.init(model, *args, transformation_dict=transformation_dict, add_methods=add_methods, **kwargs)
    
    @classmethod
    def _add_attrs_to(cls, obj, attr_names=None):
        attr_names = attr_names or ['load_weights', 'calibrate', 'freeze', 'unfreeze']
        OptimizationBaseModule._add_attrs_to(obj, attr_names)
    
    def load_weights(self, *args, **kwargs):
        quant_func_wrapper.load_weights(self.module, *args, **kwargs)

    def train(self, *args, **kwargs):
        # return quant_func.train(self.module, *args, **kwargs)
        self.module = quant_func_wrapper.train(self.module, *args, transformation_dict=self.transformation_dict, **kwargs)
        return self
    
    def calibrate(self, *args, **kwargs):
        return quant_func_wrapper.calibrate(self.module, *args, **kwargs)

    def freeze(self, *args, **kwargs):
        # return quant_func.freeze(self.module, *args, **kwargs)
        
        self.module = quant_func_wrapper.freeze(self.module, *args, transformation_dict=self.transformation_dict, **kwargs)
        return self
    

    def unfreeze(self, *args, **kwargs):
        # return quant_func.unfreeze(self.module, *args, **kwargs)  
        self.module = quant_func_wrapper.unfreeze(self.module, *args, transformation_dict=self.transformation_dict, **kwargs)
        return self
    
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def convert(self, *args, **kwargs):
        # self.module = quant_func.convert(self.module, *args, **kwargs)
        self.module = quant_func_wrapper.convert(self.module, *args, transformation_dict=self.transformation_dict, **kwargs)
        return self
    
    
    def export(self, *args, **kwargs):
        self = self.convert(*args, **kwargs)
        return quant_func_wrapper.export(self, *args, transformation_dict=self.transformation_dict, is_converted=True, **kwargs)

