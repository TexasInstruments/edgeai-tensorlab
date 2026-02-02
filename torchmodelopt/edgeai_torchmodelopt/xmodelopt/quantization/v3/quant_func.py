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

import warnings
import torch
from torch.fx import GraphModule
import torch.ao.quantization
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, prepare_qat_pt2e, convert_pt2e 
	
from .... import xnn
from ... import utils
from . import qconfig_types
from . import quant_utils
from .quantizer import get_quantizer, QuantizerTypes, QuantizerAnnotationPatterns
from ...utils.helper_functions import allow_exported_model_train_eval

import copy
import os
import types 


def _model_to_device(model, device):
    if device:
        model.to(device)
        if hasattr(model, 'recompile'):
            model.recompile()
        #
    #
    return model


def _switch_batchnorm(model, training):
    # freezing the batchnorm update 
    for n in model.graph.nodes:
        # Args: input, weight, bias, running_mean, running_var, training, momentum, eps
        # We set the `training` flag to False here to freeze BN stats
        if n.target in [
            torch.ops.aten._native_batch_norm_legit.default,
            torch.ops.aten.cudnn_batch_norm.default,
            torch.ops.aten.native_batch_norm.default,
            torch.ops.aten.batch_norm.default,
        ]:
            new_args = list(n.args)
            new_args[5] = training
            n.args = tuple(new_args)
    model.recompile()  

    
def init(model, example_inputs, example_kwargs=None, is_qat=True, total_epochs=0, 
         quantizer=None, qconfig_type=None, quantizer_type=None, annotation_patterns=None,
         num_batch_norm_update_epochs=None, num_observer_update_epochs=None, 
         add_methods=True, fast_mode=False, device=None, **kwargs):

    if not total_epochs:
        raise RuntimeError("total_epochs must be provided")
    
    if hasattr(model, '__quant_params__'):
        print('IGNORED: quant init called on a model that was already quantized \n\n\n')
        return model
    
    #####################################################################################
    # native xnnpack quantizer from Pytorch
    # from torch.ao.quantization.quantizer.xnnpack_quantizer import (get_symmetric_quantization_config, XNNPACKQuantizer)
    # quantizer = XNNPACKQuantizer()
    # quantizer.set_global(get_symmetric_quantization_config(is_qat=True))

    # see the supported values in QuantizerTypes, QuantizerAnnotationPatterns and qconfig_types.QConfigType
    quantizer_type = quantizer_type or QuantizerTypes.TIDLRT_ADVANCED
    qconfig_type = qconfig_type or qconfig_types.QConfigType.DEFAULT
    annotation_patterns = QuantizerAnnotationPatterns.DEFAULT if annotation_patterns is None else annotation_patterns
    
    # our configurable quantizer
    quantizer = quantizer or get_quantizer(quantizer_type=quantizer_type, is_qat=is_qat, fast_mode=fast_mode, device=device, annotation_patterns=annotation_patterns)

    # our configurable qconfig_type
    qconfig = qconfig_types.get_qconfig(qconfig_type, is_qat=is_qat, fast_mode=fast_mode)
    quantizer.set_global(qconfig)
    
    #####################################################################################
    example_kwargs = example_kwargs or {}
    
    if not (hasattr(model, '_example_inputs') and hasattr(model, '_example_kwargs')):
        utils.add_example_args_kwargs(model, example_inputs=example_inputs, example_kwargs=example_kwargs)
        
    example_inputs = model._example_inputs.pop(0)
    example_kwargs = model._example_kwargs.pop(0)

    if device:
        if isinstance(example_inputs, (list, tuple)):
            for i, inp in enumerate(example_inputs):
                example_inputs[i].to(device=device)
        else:
            example_inputs = [example_inputs.to(device=device)]
        for key, value in example_kwargs.items():
            if isinstance(value, torch.Tensor):
                example_kwargs[key] = value.to(device=device)
                
        model = _model_to_device(model, device)

    #####################################################################################
    orig_model =  copy.deepcopy(model) if kwargs.get('with_deepcopy', False) else model
    check_guards = kwargs.get('check_guards', True)
    example_inputs = tuple(example_inputs)
    m = torch.export.export(orig_model, example_inputs, kwargs=example_kwargs).module(check_guards=check_guards)
    from ...utils.helper_functions import allow_exported_model_train_eval
    allow_exported_model_train_eval(m)
    
    # for copy_arg in copy_args:
    #     if hasattr(module, copy_arg):
    #         setattr(replace_obj, copy_arg, getattr(module, copy_arg))
    
    #####################################################################################
    if is_qat:
        model = prepare_qat_pt2e(m, quantizer)
    else:
        model = prepare_pt2e(m, quantizer)
    
    # model = _model_to_device(model, device)

    #####################################################################################
    model.__quant_params__ = xnn.utils.AttrDict()
    model.__quant_params__.is_qat = is_qat
    model.__quant_params__.quantizer = quantizer 
    model.__quant_params__.qconfig_type = qconfig_type
    model.__quant_params__.num_batch_norm_update_epochs = num_batch_norm_update_epochs
    model.__quant_params__.num_observer_update_epochs = num_observer_update_epochs
    model.__quant_params__.num_epochs_tracked = 0
    model.__quant_params__.total_epochs = total_epochs
    model.__quant_params__.outlier_hooks = []
    model.__quant_params__.bias_hooks = []
    model.__quant_params__.bias_calibration_factor = kwargs.get("bias_calibration_factor", 0.05)
    model.__quant_params__.original_model = orig_model
    model.__quant_params__.device = device

    if add_methods:
        # add a wrapper for model.train()
        # the accuracy for deit was going down due to this wrapper, needs to be implemented properly or debugged #TODO
        # def train_quant(self, mode=True):
        #     if mode:
        #         torch.ao.quantization.move_exported_model_to_train(self)
        #     else:
        #         torch.ao.quantization.move_exported_model_to_eval(self)
                
        model.__quant_train_backup__ =  model.train.__func__
        model.train = types.MethodType(train, model)
        model.eval = types.MethodType(eval, model)
        # other methods
        model.freeze = types.MethodType(freeze, model)
        model.unfreeze = types.MethodType(unfreeze, model)
        model.convert = types.MethodType(convert, model)
        model.export = types.MethodType(export, model)
        if kwargs.get('with_deepcopy', False):
            model.__deepcopy__ = types.MethodType(deepcopy_graphmodule, model)
    #
    # this is based on module hooks - it will not work currently in pt2e
    # all modules exect the fakequant/observers in torch.export.export() model are changed to ops
    # but module hooks on fakequant/observers will work and it may be possible to change the implementation that way
    model = insert_all_hooks(model, kwargs.get('outlier_clipping',False), kwargs.get('bias_calibration',False))
    print("Model Preparation is now complete! ")
    return model


def insert_all_hooks(model, outlier_clipping, bias_calibration):
    if len(model.__quant_params__.outlier_hooks)==0 and outlier_clipping:
        model.__quant_params__.outlier_hooks += quant_utils.add_fc_outlier_supression_hook(model)
    if len(model.__quant_params__.bias_hooks)==0 and bias_calibration:
        model.__quant_params__.bias_hooks += quant_utils.add_bias_calibration_hook(model, \
                calibration_factor = model.__quant_params__.bias_calibration_factor)
    return model


def remove_hooks(hooks):
    for hook_handle in hooks:
        hook_handle.remove()
    hooks = []
    return hooks


def freeze(self, freeze_bn=True, freeze_observers=True):
    # freezing or unfreezing the observers
    for name, mod in self.named_modules():
        if hasattr(mod, "freeze_observer"):
            mod.freeze_observer = freeze_observers
    
    # this does not work, neither causes any harm, just here for future
    if freeze_observers:
        self.apply(torch.ao.quantization.disable_observer)
    else:
        self.apply(torch.ao.quantization.enable_observer)
    
    # this does not work, neither causes any harm, just here for future
    if freeze_bn is True:
        self.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        # TODO: check if this is causing accuracy degradation
        torch.ao.quantization.move_exported_model_to_eval(self)  
        # _switch_batchnorm(self, training=False)
    elif freeze_bn is False:
        self.apply(torch.nn.intrinsic.qat.update_bn_stats)
        torch.ao.quantization.move_exported_model_to_train(self)
        # _switch_batchnorm(self, training=True)

    # device = self.__quant_params__.device
    # if device:
    #     _model_to_device(self, self.__quant_params__.device)
    # #
    return self


def unfreeze(self, freeze_bn=False, freeze_observers=False):
    freeze(self, freeze_bn, freeze_observers)
    return self


def forward(self, *input, **kwargs):
    return self(*input, **kwargs)


def deepcopy_graphmodule(gm, memo=None):
    """Deep copies a GraphModule."""

    # Create a new GraphModule
    fake_mod = torch.nn.Module()
    for key in fake_mod.__dict__.keys():
        try:
            k_val = copy.deepcopy(gm.__dict__[key]) 
        except:
            k_val = {}
            for k_item in gm.__dict__[key].keys():
                try:
                    f = copy.deepcopy(gm.__dict__[key][k_item])
                except:
                    f = torch.tensor(gm.__dict__[key][k_item].item(), device=gm.__dict__[key][k_item].device)
                k_val[k_item] = f
        fake_mod.__dict__[key] = k_val
    new_gm = GraphModule(fake_mod, copy.deepcopy(gm.graph), gm.__class__.__name__)

    # Deep copy the parameters and buffers
    for name, param in gm.named_parameters():
        new_gm.register_parameter(name, copy.deepcopy(param))

    for name, buffer in gm.named_buffers():
        try:
            buf = copy.deepcopy(buffer)
        except:
            buf = torch.tensor(buffer.item(), device=buffer.device)
        if "." in name:
            mod_list = name.split(".")[:-1]
            mod_name = name.split(".")[-1]
            module = new_gm
            for mod in mod_list:
                module = getattr(module, mod)
            module.register_buffer(mod_name, buf)
        else:
            new_gm.register_buffer(name, buf)

    new_gm.meta = copy.deepcopy(gm.meta)
    return new_gm


def _convert_layers(module, fq_type, new_type):
    fq_layers = {n:m for n, m in module.named_modules(remove_duplicate=False) if isinstance(m,fq_type)}
    for n, m in fq_layers.items():
        if isinstance(m, fq_type):
            new_layer = torch.nn.Identity()
            if new_type == torch.nn.Hardtanh and hasattr(m, 'activation_post_process'):
                if hasattr(m.activation_post_process, 'get_min_max'):
                    min_val, max_val, range_valid = m.activation_post_process.get_min_max()
                else:
                    min_val, max_val = m.activation_post_process.min_val, m.activation_post_process.max_val
                    range_valid = True
                #
                if range_valid:
                    new_layer = torch.nn.Hardtanh(min_val=min_val.item(), max_val=max_val.item())
                #
            #
            parent_module = xnn.utils.get_parent_module(module, m)
            if parent_module:
                child_name = n.split('.')[-1]
                setattr(parent_module, child_name, new_layer)
            #
        #
    #
    return module


def convert(self, *args, device="cpu", make_copy=False, fq_to_clip=None, **kwargs):
    if hasattr(self, '__quant_params__'):
        orig_quant_params = self.__quant_params__
        fq_to_clip = (self.__quant_params__.qconfig_type == qconfig_types.QConfigType.WF_AFCLIP) if fq_to_clip is None else fq_to_clip
    else:
        warnings.warn("WARNING: __quant_params__ is missing in quant_func module.")
        orig_quant_params = None

    model = copy.deepcopy(self) if make_copy else self # calls the deepcopy_graphmodule module
    model = model.to(device=device)
    model = quant_utils.move_node_kwargs_to_device(model, device=device)
    model = quant_utils.remove_to_device_node(model)

    if fq_to_clip:
        model = _convert_layers(model, torch.ao.quantization.FakeQuantize, torch.nn.Hardtanh)
        model = _convert_layers(model, torch.ao.quantization.observer.PlaceholderObserver, torch.nn.Identity)
        model.graph.lint()
        model.recompile()
    else:
        # just to make legacy code in convert_pt2e happy
        # see: torch/ao/quantization/pt2e/qat_utils.py _fold_conv_bn_qat()
        for node in model.graph.nodes:
            if "source_fn_stack" not in node.meta:
                node.meta["source_fn_stack"] = []
        #
        # now actually convert the model
        model = convert_pt2e(model)
    
    # torch.ao.quantization.move_exported_model_to_eval(model)
    model.train = types.MethodType(train, model)
    model.eval = types.MethodType(eval, model)

    if orig_quant_params:
        setattr(model, "__quant_params__", orig_quant_params)
    
    return model


def train(self, mode: bool = True):
    # hf transformers call train before every iteration, which messes with epochs tracked, will be needing a better logic for that #TODO
    # as of now, pass the expected number of epochs in num_batch_norm_update_epochs and num_observer_update_epochs variables, same is expected 
    # if the training code calls model.train() before every iteration 
    # put the model in expected mode
    if hasattr(self, "__quant_train_backup__"):
        self.__quant_train_backup__(mode=mode)
    # also freeze the params if required
    if mode is True:
        # set the default epoch at which freeze occurs during training (if missing)
        num_batch_norm_update_epochs = ((self.__quant_params__.total_epochs//2)-1) \
            if self.__quant_params__.num_batch_norm_update_epochs is None else int(self.__quant_params__.num_batch_norm_update_epochs)
        
        num_observer_update_epochs = ((self.__quant_params__.total_epochs//2)+1) \
            if self.__quant_params__.num_observer_update_epochs is None else int(self.__quant_params__.num_observer_update_epochs)
        num_observer_update_epochs = max(num_observer_update_epochs, 1) # must run observer for atleast 1 epoch - otherwise range will not be availble and fake_quatize will have errors.

        freeze_bn = (self.__quant_params__.num_epochs_tracked >= num_batch_norm_update_epochs)
        freeze_observers = (self.__quant_params__.num_epochs_tracked >= num_observer_update_epochs)

        if freeze_bn:
            xnn.utils.print_once('INFO: Freezing BN for subsequent quantization epochs')
        #
        if freeze_observers:
            xnn.utils.print_once('INFO: Freezing ranges for subsequent quantization epochs')
        #
        freeze(self, freeze_bn=freeze_bn, freeze_observers=freeze_observers)
        
        # TODO: we will probably need better logic to extend to adding more hooks in the toolkit #TODO
        # if len(self.__quant_params__.outlier_hooks)==0 and not(freeze_observers):
        #     self = insert_all_hooks(self, outlier_clipping=True, bias_calibration=False)
        # if len(self.__quant_params__.bias_hooks)==0:
        #     self = insert_all_hooks(self, outlier_clipping=False, bias_calibration=True)

        # Removing the outlier hook when the observers are also frozen
        if freeze_observers and len(self.__quant_params__.outlier_hooks)>0:
            self.__quant_params__.outlier_hooks = remove_hooks(self.__quant_params__.outlier_hooks)
          
        self.__quant_params__.num_epochs_tracked += 1
    else:
        # TODO: add later as required
        # self.__quant_params__.bias_hooks = remove_hooks(self.__quant_params__.bias_hooks)                      
        # self.__quant_params__.outlier_hooks = remove_hooks(self.__quant_params__.outlier_hooks)
        freeze(self)
    #
    return self


def eval(self, mode: bool = False):
    return train(self, mode)


def calibrate(self, freeze_bn=True, freeze_observers=False, freeze_fn=None):
    # self.eval()
    freeze_fn=freeze_fn or freeze
    freeze_fn(self, freeze_bn, freeze_observers)
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

    
# from: torch/ao/quantization/fx/graph_module.py
def _is_observed_module(module) -> bool:
    return hasattr(module, "meta") and "_observed_graph_module_attrs" in module.meta


def export(self, example_inputs, filename='model.onnx', opset_version=17, model_qconfig_format=None, preserve_qdq_model=True,
           simplify=True, skipped_optimizers=None, device='cpu', make_copy=True, insert_metadata=True, is_converted=False, **export_kwargs):

    if _is_observed_module(self):
        model = convert(self, device=device, make_copy=make_copy)
    elif not is_converted:
        model = convert(self, device=device, make_copy=make_copy)
    else:
        model = self
        warnings.warn("model has already been converted before calling export. make sure it is done correctly.")

    model.module = quant_utils.remove_loss_branch(model.module)
    quant_utils.register_onnx_symbolics()

    if model_qconfig_format == qconfig_types.QConfigFormat.INT_MODEL:
        # # Convert QDQ format to Int8 format
        import onnxruntime as ort
        qdq_filename = os.path.splitext(filename)[0] + '_qdq.onnx'
        torch.onnx.export(model, example_inputs.to('cpu'), qdq_filename, opset_version=opset_version, training=torch._C._onnx.TrainingMode.PRESERVE, **export_kwargs)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        so.optimized_model_filepath = filename
        # logger.info("Inplace conversion of QDQ model to INT8 model at: {}".format(onnx_file))
        ort.InferenceSession(qdq_filename, so)
        if not preserve_qdq_model:
            os.remove(qdq_filename)
        #
    else:
        if isinstance(example_inputs, dict):
            input_to_export = ()
            for val in example_inputs.values():
                if isinstance(val, list):
                    continue
                input_to_export += tuple([val.to(device=device)])
        else:
            input_to_export = example_inputs.to(device=device)
        torch.onnx.export(model, input_to_export, filename, opset_version=opset_version, training=torch._C._onnx.TrainingMode.PRESERVE, **export_kwargs)

    if simplify:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(filename)
        onnx_model, check = simplify(onnx_model, skipped_optimizers=skipped_optimizers)
        onnx.save(onnx_model, filename)
    
    if insert_metadata:
        import onnx
        from ....version import __version__
        onnx_model = onnx.load(filename)
        meta = onnx_model.metadata_props.add()
        meta.key = "model_source"
        meta.value = f"edgeai_torchmodelopt_{__version__}"
        onnx.save(onnx_model, filename)
