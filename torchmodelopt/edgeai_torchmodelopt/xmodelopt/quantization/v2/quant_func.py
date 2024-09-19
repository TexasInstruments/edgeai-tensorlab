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
import copy

from .... import xnn

from ...surgery.v2 import custom_surgery_functions, convert_to_lite_fx

from . import qconfig_types
from . import quant_utils

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
            qconfig_mode=qconfig_types.QConfigMode.DEFAULT, add_methods=True, dynamo_export=False, **kwargs):

    if hasattr(model, '__quant_params__'):
        print('IGNORED: quant init called on a model that was already quantized')
        return model
    #

    if not total_epochs:
        if not is_qat:
            total_epochs = 2
        else:
            raise RuntimeError("total_epochs must be provided")
    #

    if dynamo_export:
        example_inputs = example_inputs if example_inputs is not None else \
            torch.ones(1,3,224,224).to(next(model.parameters()).device)
        example_inputs = example_inputs[0] if isinstance(example_inputs, tuple) else example_inputs

        if kwargs.get('convert_to_cuda', False):
            for key, value in example_inputs.items():
                example_inputs[key] = value.to(device='cuda:0')
            model = model.to(device='cuda:0')

        #orig_model = copy.deepcopy(model)

        if isinstance(example_inputs, dict):
            m, guards = torchdynamo.export(model, **example_inputs, aten_graph=False, assume_static_by_default=True)
            print("Dynamo Export Completed ! \n\n")
        else:
            m, guards = torchdynamo.export(model, example_inputs, aten_graph=False, assume_static_by_default=True)
        #
        model = m
		
    
    if has_timm:
        replacement_dict={
            "attention_to_quant_attention": {models.vision_transformer.Attention: quant_utils.QuantAttention},
            "window_attention_to_quant_attention": {models.swin_transformer.WindowAttention: quant_utils.QuantAttention},
            "layer_norm_to_quant_layer_norm": {nn.LayerNorm: quant_utils.QuantLayerNorm},
            "permute_change_to_export":{'permute': custom_surgery_functions.replace_permute_layer}
        }
    else:
        replacement_dict={
            "layer_norm_to_quant_layer_norm": {nn.LayerNorm: quant_utils.QuantLayerNorm},
            "permute_change_to_export":{'permute': custom_surgery_functions.replace_permute_layer}
        }
        
    orig_device = next(model.parameters()).device
    copy_args=["scale", "qkv", "proj", "num_heads", "head_dim", "weight", "bias", "eps",
                "relative_position_index", "relative_position_bias_table", "window_area"]
    model = convert_to_lite_fx(model, replacement_dict, copy_args=copy_args)
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
    if is_qat:
        model = quantize_fx.prepare_qat_fx(model, qconfig_mapping, example_inputs)
    else:
        model.eval()
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
    model.__quant_params__.outlier_hooks = []
    model.__quant_params__.bias_hooks = []
    model.__quant_params__.bias_calibration_factor = kwargs.get("bias_calibration_factor", 0)
    model = qconfig_types.adjust_matmul_inputs_qconfig(model)

    if add_methods:
        # add a wrapper for model.train()
        model.__train_backup__ = types.MethodType(model.train.__func__, model)
        model.train = types.MethodType(train, model)
        model.eval = types.MethodType(train, model)
        # other methods
        model.freeze = types.MethodType(freeze, model)
        model.unfreeze = types.MethodType(unfreeze, model)
        model.convert = types.MethodType(convert, model)
        model.export = types.MethodType(export, model)
    #    
    print("Model Preparation is now complete! ")
    
    model = insert_all_hooks(model)
    return model


def insert_all_hooks(model, insert_outlier_hook=True, insert_bias_hook = True):
    # if len(model.__quant_params__.outlier_hooks)==0 and insert_outlier_hook:
    #     model.__quant_params__.outlier_hooks += quant_utils.add_fc_outlier_supression_hook(model)
    if len(model.__quant_params__.bias_hooks)==0 and insert_bias_hook:
        model.__quant_params__.bias_hooks += quant_utils.add_bias_calibration_hook(model, \
                calibration_factor = model.__quant_params__.bias_calibration_factor)
    return model


def remove_hooks(hooks):
    for hook_handle in hooks:
        hook_handle.remove()
    hooks = []
    return hooks
    
    
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


def convert(self, device='cpu', model_quant_format=None, convert_custom_config=None, backend_config=None, make_copy=False):
    orig_quant_params = copy.deepcopy(self.__quant_params__)
    model = copy.deepcopy(self).eval() if make_copy else self.eval()
    model = model.to(device=device)
    # now do the actual conversion
    model = quantize_fx.convert_fx(model, convert_custom_config=convert_custom_config, backend_config=backend_config)
    model.eval = types.MethodType(train, model)
    setattr(model, "__quant_params__", orig_quant_params)
    return model


def export(self, example_input, filename='model.onnx', opset_version=17, model_quant_format=None, preserve_qdq_model=True,
           simplify=True, skipped_optimizers=None, device='cpu', make_copy=True, is_converted=False, **export_kwargs):
    if not is_converted:
        model = convert(self, device=device, make_copy=make_copy)
    else:
        model = self
        
    register_custom_op_symbolic(
        symbolic_name='quantized::matmul', 
        symbolic_fn=quant_utils.quantized_matmul, 
        opset_version=17)

    register_custom_op_symbolic(
        symbolic_name='quantized::softmax', 
        symbolic_fn=quant_utils.quantized_softmax, 
        opset_version=17)
    
    if model_quant_format == ModelQuantFormat.INT_MODEL:
        # # Convert QDQ format to Int8 format
        import onnxruntime as ort
        qdq_filename = os.path.splitext(filename)[0] + '_qdq.onnx'
        torch.onnx.export(model, example_input.to(device=device), qdq_filename, opset_version=opset_version, **export_kwargs)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        so.optimized_model_filepath = filename
        # logger.info("Inplace conversion of QDQ model to INT8 model at: {}".format(onnx_file))
        ort.InferenceSession(qdq_filename, so)
        if not preserve_qdq_model:
            os.remove(qdq_filename)
        #
    else:
        torch.onnx.export(model, example_input.to(device=device), filename, opset_version=opset_version, **export_kwargs)
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
        
        # we will probably need better logic to extend to adding more hooks in the toolkit #TODO
        if len(self.__quant_params__.outlier_hooks)==0 and not(freeze_observers):
            self = insert_all_hooks(self, insert_bias_hook=False)
        if len(self.__quant_params__.bias_hooks)==0:
            self = insert_all_hooks(self, insert_outlier_hook=False)
        
        # Removing the outlier hook when the observers are also frozen
        if freeze_observers and len(self.__quant_params__.outlier_hooks)>0:
            self.__quant_params__.outlier_hooks = remove_hooks(self.__quant_params__.outlier_hooks)
        
        self.__quant_params__.num_epochs_tracked += 1
    else:
        self.__quant_params__.bias_hooks = remove_hooks(self.__quant_params__.bias_hooks)                      
        self.__quant_params__.outlier_hooks = remove_hooks(self.__quant_params__.outlier_hooks)
        freeze(self)
    #
    return self


def calibrate(self, freeze_bn=True, freeze_observers=False, freeze_fn=None):
    self.eval()
    freeze_fn=freeze_fn or freeze
    freeze_fn(self, freeze_bn, freeze_observers)
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
