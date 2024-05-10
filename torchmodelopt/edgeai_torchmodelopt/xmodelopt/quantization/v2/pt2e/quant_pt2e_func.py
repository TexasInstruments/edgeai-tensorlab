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
import torch._dynamo as torchdynamo
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, prepare_qat_pt2e, convert_pt2e 

from ..... import xnn
from . import qconfig_types
from . import quant_pt2e_utils
from .quantizers import TIDLRTQuantizer

import copy
import os
import types

class ModelQuantFormat:
    FLOAT_MODEL = "FLOAT_MODEL"
    FAKEQ_MODEL = "FAKEQ_MODEL"
    QDQ_MODEL = "QDQ_MODEL"
    INT_MODEL = "INT_MODEL"
    _NUM_FORMATS_ = 4


def init(model, quantizer=None, is_qat=True, total_epochs=0, example_inputs=None, qconfig_type=None,
         qconfig_mode=qconfig_types.QConfigMode.DEFAULT,num_batch_norm_update_epochs=None, 
         num_observer_update_epochs=None, add_methods=True, **kwargs):
    
    if hasattr(model, '__quant_params__'):
        print('IGNORED: quant init called on a model that was already quantized')
        return model
    
    if not total_epochs:
        if not is_qat:
            total_epochs = 1
        else:
            raise RuntimeError("total_epochs must be provided")

    example_inputs = example_inputs if example_inputs is not None else \
        torch.ones(1,3,224,224).to(next(model.parameters()).device)
    example_inputs = example_inputs[0] if isinstance(example_inputs, tuple) else example_inputs

    if isinstance(example_inputs, dict):
        m, guards = torchdynamo.export(model, **example_inputs, aten_graph=True, assume_static_by_default=True)
    else:
        m, guards = torchdynamo.export(model, example_inputs, aten_graph=True, assume_static_by_default=True)
    
    qconfig_type = qconfig_type or qconfig_types.QConfigType.DEFAULT
    qconfig_mode = qconfig_types.get_qconfig(qconfig_type)
    
    # methods to quantize individual layers/modules types are in quantizer
    quantizer = quantizer or TIDLRTQuantizer()
    quantizer.set_global(qconfig_mode)
    
    if is_qat:
        model = prepare_qat_pt2e(m, quantizer)
    else:
        model = prepare_pt2e(m, quantizer)
        
    # TODO torch 2.3 test this 
    # model = model.module()
    
    model.__quant_params__ = xnn.utils.AttrDict()
    model.__quant_params__.is_qat = is_qat
    model.__quant_params__.quantizer = quantizer # maybe remove this only, not needed
    model.__quant_params__.num_batch_norm_update_epochs = num_batch_norm_update_epochs
    model.__quant_params__.num_observer_update_epochs = num_observer_update_epochs
    model.__quant_params__.num_epochs_tracked = 0
    model.__quant_params__.total_epochs = total_epochs
    
    # related to adaptive quantization
    # model.__quant_params__.qconfig_mode = qconfig_types.QConfigMode(qconfig_mode)
    # model.__quant_params__.forzen_layer_names_list = []
    
    model = qconfig_types.adjust_mixed_precision_qconfig(model, is_qat, qconfig_type)

    
    if add_methods:
        # add a wrapper for model.train()
        # model.__train_backup__ = types.MethodType(model.train.__func__, model)
        model.train = types.MethodType(train, model)
        model.eval = types.MethodType(train, model)
        # other methods
        model.freeze = types.MethodType(freeze, model)
        model.unfreeze = types.MethodType(unfreeze, model)
        model.convert = types.MethodType(convert, model)
        model.export = types.MethodType(export, model)
    #
    print("Model Preparation is now complete! ")
    return model


def freeze(self, freeze_bn=True, freeze_observers=True):
    if freeze_observers is True:
        self.apply(torch.ao.quantization.disable_observer)
    elif freeze_observers is False:
        self.apply(torch.ao.quantization.enable_observer)
    #
    if freeze_bn is True:
        self.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    elif freeze_bn is False:
        self.apply(torch.nn.intrinsic.qat.update_bn_stats)
    #
    return self


def unfreeze(self, freeze_bn=False, freeze_observers=False):
    freeze(self, freeze_bn, freeze_observers)
    return self


def forward(self, *input, **kwargs):
    return self(*input, **kwargs)


def convert(self, device='cpu'):
    freeze(self)
    # convert requires cpu model 
    #TODO check of this is required
    # self.to(torch.device(device))
    # now do the actual conversion
    self = convert_pt2e(self)
    return self


def train(self, mode: bool = True):
    # put the model in expected mode
    if hasattr(self, "__train_backup__"):
        self.__train_backup__(mode=mode)
    # also freeze the params if required
    if mode is True:
        # set the default epoch at which freeze occurs during training (if missing)
        num_batch_norm_update_epochs = self.__quant_params__.num_batch_norm_update_epochs or ((self.__quant_params__.total_epochs//2)-1)
        num_observer_update_epochs = self.__quant_params__.num_observer_update_epochs or ((self.__quant_params__.total_epochs//2)+1)
        freeze_bn = (self.__quant_params__.num_epochs_tracked >= num_batch_norm_update_epochs)
        freeze_observers = (self.__quant_params__.num_epochs_tracked >= num_observer_update_epochs)
        if freeze_bn:
            xnn.utils.print_once('Freezing BN for subsequent epochs')
        #
        if freeze_observers:
            xnn.utils.print_once('Freezing ranges for subsequent epochs')
        #
        freeze(self, freeze_bn=freeze_bn, freeze_observers=freeze_observers)
        quant_pt2e_utils.adjust_gradual_quantization(self)
        self.__quant_params__.num_epochs_tracked += 1
    else:
        freeze(self)
    #
    return self


def calibrate(self, freeze_bn=True, freeze_observers=False):
    self.eval()
    freeze(self, freeze_bn, freeze_observers)
    return self


# dont think that it will be required, atleast for transformers, need to see in other repos
def load_weights(self, pretrained, *args, strict=True, state_dict_name=None, **kwargs):
    data_dict = torch.load(self, pretrained, *args, **kwargs)
    if state_dict_name:
        state_dict_names = state_dict_name if isinstance(state_dict_name, (list,tuple)) else [state_dict_name]
        for s_name in state_dict_names:
            data_dict = data_dict[s_name] if ((data_dict is not None) and s_name in data_dict) else data_dict
        #
    #
    self.load_state_dict(data_dict, strict=strict)
    
    
def export(self, example_input, filename='model.onnx', opset_version=17, model_quant_format=None, preserve_qdq_model=True,
           simplify=False, skipped_optimizers=None):
    
    quant_pt2e_utils.register_onnx_symbolics()
    model = copy.deepcopy(self)
    model = convert(model).to('cpu')
    model = quant_pt2e_utils.remove_loss_branch(model)    
    
    if model_quant_format == ModelQuantFormat.INT_MODEL:
        # # Convert QDQ format to Int8 format
        import onnxruntime as ort
        qdq_filename = os.path.splitext(filename)[0] + '_qdq.onnx'
        torch.onnx.export(model, example_input.to('cpu'), qdq_filename, opset_version=opset_version, training=torch._C._onnx.TrainingMode.PRESERVE)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        so.optimized_model_filepath = filename
        # logger.info("Inplace conversion of QDQ model to INT8 model at: {}".format(onnx_file))
        ort.InferenceSession(qdq_filename, so)
        if not preserve_qdq_model:
            os.remove(qdq_filename)
        #
    else:
        if isinstance(example_input, dict):
            example_inputs = ()
            for val in example_input.values():
                example_inputs += tuple([val.to('cpu')])
        else:
            example_inputs = example_input.to('cpu')
        torch.onnx.export(model, example_inputs, filename, opset_version=opset_version, training=torch._C._onnx.TrainingMode.PRESERVE)
    #
    if simplify:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(filename)
        onnx_model, check = simplify(onnx_model, skipped_optimizers=skipped_optimizers)
        onnx.save(onnx_model, filename)
    