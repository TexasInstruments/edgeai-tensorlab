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
import torch._dynamo as torchdynamo
from torch.fx import GraphModule
import torch.ao.quantization
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, prepare_qat_pt2e, convert_pt2e 

from .... import xnn
from ... import utils
from . import qconfig_types
from . import quant_utils
from .quantizers import TIDLRTQuantizer

import copy
import os
import types 

def init(model, quantizer=None, is_qat=True, total_epochs=0, example_inputs=None, example_kwargs=None, qconfig_type=None,
        qconfig_mode=qconfig_types.QConfigMode.DEFAULT, num_batch_norm_update_epochs=None, num_observer_update_epochs=None, 
        add_methods=True, fast_mode=False, is_fake_quantize=True, **kwargs):
    
    if hasattr(model, '__quant_params__'):
        print('IGNORED: quant init called on a model that was already quantized \n\n\n')
        return model
    
    example_kwargs = example_kwargs or {} 
    if hasattr(model, '_example_inputs') and hasattr(model, '_example_kwargs'):
        example_inputs = model._example_inputs.pop(0)
        example_kwargs = model._example_kwargs.pop(0)
    else:
        utils.add_example_args_kwargs(model, example_inputs=example_inputs, example_kwargs=example_kwargs)
    
    if not total_epochs:
        if not is_qat:
            total_epochs = 2
        else:
            raise RuntimeError("total_epochs must be provided")

    example_inputs = example_inputs if example_inputs is not None else \
        torch.ones(1,3,224,224).to(next(model.parameters()).device)
    
    if kwargs.get('convert_to_cuda', False):
        if isinstance(example_inputs, (list, tuple)):
            for i, inp in enumerate(example_inputs):
                example_inputs[i] = example_inputs[i].to(device='cuda:0')
        else:
            example_inputs = [example_inputs.to(device='cuda:0')]
        for key, value in example_kwargs.items():
            if isinstance(value, torch.Tensor):
                example_kwargs[key] = value.to(device='cuda:0')
        model = model.to(device='cuda:0')

    orig_model = copy.deepcopy(model)
        
    decomposition_table = {torch.ops.aten.layer_norm.default: quant_utils.native_layer_norm}
    
    m, guards = torchdynamo.export(orig_model, aten_graph=True, assume_static_by_default=True, pre_dispatch=True, decomposition_table=decomposition_table)(*example_inputs, **example_kwargs)
    print("Dynamo Export Completed ! \n\n")
    
    is_fake_quantize = True if is_qat else is_fake_quantize
    qconfig_type = qconfig_type or qconfig_types.QConfigType.DEFAULT
    qconfig_mode = qconfig_types.get_qconfig(qconfig_type, is_fake_quantize=is_fake_quantize, fast_mode=fast_mode)
    
    # methods to quantize individual layers/modules types are in quantizer
    quantizer = quantizer or TIDLRTQuantizer(is_qat=is_qat, fast_mode=fast_mode, is_fake_quantize=is_fake_quantize, device=next(iter(m.named_parameters()))[1].device)
    quantizer.set_global(qconfig_mode)
    
    # for copy_arg in copy_args:
    #     if hasattr(module, copy_arg):
    #         setattr(replace_obj, copy_arg, getattr(module, copy_arg))
    
    if kwargs.get('convert_to_cuda', False):
        m = m.to(device='cpu')
    
    if is_qat:
        model = prepare_qat_pt2e(m, quantizer)
    else:
        model = prepare_pt2e(m, quantizer)
        
    # TODO torch 2.3 test this 
    # model = model.module()
    
    model.__quant_params__ = xnn.utils.AttrDict()
    model.__quant_params__.is_qat = is_qat
    model.__quant_params__.quantizer = quantizer 
    model.__quant_params__.num_batch_norm_update_epochs = num_batch_norm_update_epochs
    model.__quant_params__.num_observer_update_epochs = num_observer_update_epochs
    model.__quant_params__.num_epochs_tracked = 0
    model.__quant_params__.total_epochs = total_epochs
    model.__quant_params__.outlier_hooks = []
    model.__quant_params__.bias_hooks = []
    model.__quant_params__.bias_calibration_factor = kwargs.get("bias_calibration_factor", 0)
    model.__quant_params__.original_model = orig_model

    if add_methods:
        # add a wrapper for model.train()
        # the accuracy for deit was going down due to this wrapper, needs to be implemented properly or debugged #TODO
        # def train_quant(self, mode=True):
        #     if mode:
        #         torch.ao.quantization.move_exported_model_to_train(self)
        #     else:
        #         torch.ao.quantization.move_exported_model_to_eval(self)
                
        # model.__quant_train_backup__ = types.MethodType(train_quant if is_qat else model.train.__func__, model)
        model.train = types.MethodType(train, model)
        model.eval = types.MethodType(eval, model)
        # other methods
        model.freeze = types.MethodType(freeze, model)
        model.unfreeze = types.MethodType(unfreeze, model)
        model.convert = types.MethodType(convert, model)
        model.export = types.MethodType(export, model)
        model.__deepcopy__ = types.MethodType(deepcopy_graphmodule, model)
    #
    print("Model Preparation is now complete! ")
    
    model = insert_all_hooks(model)
    return model


def insert_all_hooks(model, insert_outlier_hook=True, insert_bias_hook = True):
    if len(model.__quant_params__.outlier_hooks)==0 and insert_outlier_hook:
        model.__quant_params__.outlier_hooks += quant_utils.add_fc_outlier_supression_hook(model)
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
    elif freeze_bn is False:
        self.apply(torch.nn.intrinsic.qat.update_bn_stats)
    
    # freezing the batchnorm update 
    for n in self.graph.nodes:
        # Args: input, weight, bias, running_mean, running_var, training, momentum, eps
        # We set the `training` flag to False here to freeze BN stats
        if n.target in [
            torch.ops.aten._native_batch_norm_legit.default,
            torch.ops.aten.cudnn_batch_norm.default,
            torch.ops.aten.batch_norm.default,
        ]:
            new_args = list(n.args)
            new_args[5] = not(freeze_bn)
            n.args = tuple(new_args)
    self.recompile()      
            
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


def convert(self, *args, device="cpu", make_copy=True, **kwargs):
    if hasattr(self, '__quant_params__'):
        orig_quant_params = copy.deepcopy(self.__quant_params__)
    else:
        warnings.warn("__quant_params__ is missing in quant_func module. it may be due to a deepcopy.")
        orig_quant_params = None

    model = copy.deepcopy(self).eval() if make_copy else self.eval() # calls the deepcopy_graphmodule module
    model = model.to(device=device)
    model = quant_utils.move_node_kwargs_to_device(model, device=device)
    model = quant_utils.remove_to_device_node(model)
    model = convert_pt2e(model, use_reference_representation=False, fold_quantize= False)
    model.eval = types.MethodType(eval, model)
    torch.ao.quantization.move_exported_model_to_eval(model)
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
        # torch.ao.quantization.move_exported_model_to_train(self)
        # set the default epoch at which freeze occurs during training (if missing)
        num_batch_norm_update_epochs = self.__quant_params__.num_batch_norm_update_epochs or ((self.__quant_params__.total_epochs//2)-1)
        num_observer_update_epochs = self.__quant_params__.num_observer_update_epochs or ((self.__quant_params__.total_epochs//2)+1)
        freeze_bn = (self.__quant_params__.num_epochs_tracked >= num_batch_norm_update_epochs)
        freeze_observers = (self.__quant_params__.num_epochs_tracked >= num_observer_update_epochs)
        # freeze_bn = freeze_observers = False      ####TODO WHY turned off?? FIXME
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
        # torch.ao.quantization.move_exported_model_to_eval(self) # causing accuracy degradation
        freeze(self)
    #
    return self


def eval(self, mode: bool = False):
    return train(self, mode)


def calibrate(self, freeze_bn=True, freeze_observers=False, freeze_fn=None):
    self.eval()
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
