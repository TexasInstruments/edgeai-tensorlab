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
import os
import torch
import torch.nn as nn
from torch.ao.quantization import quantize_fx
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization import FakeQuantize
from torch.ao.quantization import get_default_qconfig
import statistics
import functools
import types
from torch.onnx import register_custom_op_symbolic


from ..... import xnn

from ....surgery.v2 import custom_surgery_functions, replace_unsupported_layers

from ...surgery.v2 import custom_surgery_functions, replace_unsupported_layers

from . import qconfig_types
from . import quant_fx_utils

try:
    from timm import models
    has_timm = True
except:
    has_timm = False

class ModelQuantFormat:
    FLOAT_MODEL = "FLOAT_MODEL"
    FAKEQ_MODEL = "FAKEQ_MODEL"
    QDQ_MODEL = "QDQ_MODEL"
    INT_MODEL = "INT_MODEL"
    _NUM_FORMATS_ = 4


def init(model, qconfig_type=None, example_inputs=None, is_qat=True, backend="qnnpack",
                 total_epochs=0, num_batch_norm_update_epochs=None, num_observer_update_epochs=None,
                 qconfig_mode=qconfig_types.QConfigMode.DEFAULT, add_methods=True, **kwargs):

    if hasattr(model, '__quant_params__'):
        print('IGNORED: quant init called on a model that was already quantized')
        return model
    #

    if not total_epochs:
        raise RuntimeError("total_epochs must be provided")
    #
    
    if has_timm:
        replacement_dict={
            models.vision_transformer.Attention : quant_fx_utils.QuantAttention,
            models.swin_transformer.WindowAttention : quant_fx_utils.QuantAttention,
            nn.LayerNorm : quant_fx_utils.QuantLayerNorm,
            'permute' : custom_surgery_functions.replace_permute_layer
        }
    else:
        replacement_dict={
            nn.LayerNorm : quant_fx_utils.QuantLayerNorm,
            'permute' : custom_surgery_functions.replace_permute_layer
        }
        
    orig_device = next(model.parameters()).device
    copy_args=["scale", "qkv", "proj", "num_heads", "head_dim", "weight", "bias", "eps",
                "relative_position_index", "relative_position_bias_table", "window_area"]
    model = replace_unsupported_layers(model, replacement_dict, copy_args=copy_args)
    model = model.to(orig_device)

    # handle None here
    qconfig_type = qconfig_type or qconfig_types.QConfigType.DEFAULT
    # split based on + for mixed precision
    qconfig_type = qconfig_type.split("+") if isinstance(qconfig_type, str) else (qconfig_type,)
    if len(qconfig_type) > 2:
        raise RuntimeError(f"maximum of 2 entries are supported in qconfig_type:{qconfig_type}")
    #

    # set the quantization backend - qnnpack, fbgemm, x86, onednn etc.
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported: " + str(backend))
    torch.backends.quantized.engine = backend

    qconfig_mapping = qconfig_types.get_qconfig_mapping(is_qat, backend, qconfig_type)
    qconfig_mapping1 = QConfigMapping().set_global(get_default_qconfig("qnnpack"))
    if is_qat:
        model = quantize_fx.prepare_qat_fx(model, qconfig_mapping, example_inputs)
    else:
        model = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)
    #

    # a place to put all state variables
    model.__quant_params__ = xnn.utils.AttrDict()
    model.__quant_params__.is_qat = is_qat
    model.__quant_params__.backend = backend
    model.__quant_params__.qconfig_type = qconfig_type
    model.__quant_params__.num_batch_norm_update_epochs = num_batch_norm_update_epochs
    model.__quant_params__.num_observer_update_epochs = num_observer_update_epochs
    model.__quant_params__.num_epochs_tracked = 0
    model.__quant_params__.total_epochs = total_epochs

    # related to adaptive quantization
    model.__quant_params__.qconfig_mode = qconfig_types.QConfigMode(qconfig_mode)
    model.__quant_params__.forzen_layer_names_list = []

    model = qconfig_types.adjust_matmul_inputs_qconfig(model)
    model = qconfig_types.adjust_fc_outlier_supression(model)
    model = qconfig_types.adjust_mixed_precision_qconfig(model, is_qat, backend, qconfig_type)

    if add_methods:
        # add a wrapper for model.train()
        model.__train_backup__ = types.MethodType(model.train.__func__, model)
        model.train = types.MethodType(train, model)
        # other methods
        model.freeze = types.MethodType(freeze, model)
        model.unfreeze = types.MethodType(unfreeze, model)
        model.convert = types.MethodType(convert, model)
        model.export = types.MethodType(export, model)
    #
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


def convert(self, device='cpu', model_quant_format=None, convert_custom_config=None, backend_config=None):
    freeze(self)
    # convert requires cpu model
    self.to(torch.device(device))
    # now do the actual conversion
    self = quantize_fx.convert_fx(self, convert_custom_config=convert_custom_config, backend_config=backend_config)
    return self


def export(self, example_input, filename='model.onnx', opset_version=17, model_quant_format=None, preserve_qdq_model=True,
           simplify=False, skipped_optimizers=None):
    
    register_custom_op_symbolic(
        symbolic_name='quantized::matmul', 
        symbolic_fn=quant_fx_utils.quantized_matmul, 
        opset_version=17)

    register_custom_op_symbolic(
        symbolic_name='quantized::softmax', 
        symbolic_fn=quant_fx_utils.quantized_softmax, 
        opset_version=17)
    
    model = copy.deepcopy(self)
    model = convert(model)
    if model_quant_format == ModelQuantFormat.INT_MODEL:
        # # Convert QDQ format to Int8 format
        import onnxruntime as ort
        qdq_filename = os.path.splitext(filename)[0] + '_qdq.onnx'
        torch.onnx.export(model, example_input.to('cpu'), qdq_filename, opset_version=opset_version)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        so.optimized_model_filepath = filename
        # logger.info("Inplace conversion of QDQ model to INT8 model at: {}".format(onnx_file))
        ort.InferenceSession(qdq_filename, so)
        if not preserve_qdq_model:
            os.remove(qdq_filename)
        #
    else:
        torch.onnx.export(model, example_input.to('cpu'), filename, opset_version=opset_version)
    #
    if simplify:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(filename)
        onnx_model, check = simplify(onnx_model, skipped_optimizers=skipped_optimizers)
        onnx.save(onnx_model, filename)
    #


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
        quant_fx_utils.adjust_gradual_quantization(self)
        self.__quant_params__.num_epochs_tracked += 1
    else:
        freeze(self)
    #
    return self


def calibrate(self, freeze_bn=True, freeze_observers=False):
    self.eval()
    freeze(self, freeze_bn, freeze_observers)
    return self
    

def load_weights(self, pretrained, *args, strict=True, state_dict_name=None, **kwargs):
    data_dict = torch.load(self, pretrained, *args, **kwargs)
    if state_dict_name:
        state_dict_names = state_dict_name if isinstance(state_dict_name, (list,tuple)) else [state_dict_name]
        for s_name in state_dict_names:
            data_dict = data_dict[s_name] if ((data_dict is not None) and s_name in data_dict) else data_dict
        #
    #
    self.load_state_dict(data_dict, strict=strict)
