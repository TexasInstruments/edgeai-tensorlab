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
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, prepare_qat_pt2e, convert_pt2e 
from torch._export import capture_pre_autograd_graph

from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config

from .... import xnn
from . import qconfig_types
from . import quant_utils
from .quantizers import TIDLRTQuantizer

import copy
import os
import types


def init(model, quantizer=None, is_qat=True, total_epochs=0, example_inputs=None, qconfig_type=None,
         qconfig_mode=qconfig_types.QConfigMode.DEFAULT,num_batch_norm_update_epochs=None, 
         num_observer_update_epochs=None, add_methods=True, fast_mode=False, **kwargs):
    
    if hasattr(model, '__quant_params__'):
        print('IGNORED: quant init called on a model that was already quantized')
        return model
    
    if not total_epochs:
        if not is_qat:
            total_epochs = 2
        else:
            raise RuntimeError("total_epochs must be provided")

    example_inputs = example_inputs if example_inputs is not None else \
        torch.ones(1,3,224,224).to(next(model.parameters()).device)
    example_inputs = example_inputs[0] if isinstance(example_inputs, (list, tuple)) else example_inputs
    
    if kwargs.get('convert_to_cuda', False):
        for key, value in example_inputs.items():
            example_inputs[key] = value.to(device='cuda:0')
        model = model.to(device='cuda:0')

    orig_model = copy.deepcopy(model)
    
    if isinstance(example_inputs, dict):
        m, guards = torchdynamo.export(model, **example_inputs, aten_graph=True, assume_static_by_default=True, )
        print("Dynamo Export Completed ! \n\n")
    else:
        m, guards = torchdynamo.export(model, example_inputs, aten_graph=True, assume_static_by_default=True)
    
    qconfig_type = qconfig_type or qconfig_types.QConfigType.DEFAULT
    qconfig_mode = qconfig_types.get_qconfig(qconfig_type, is_qat=is_qat, fast_mode=fast_mode)
    
    # qconfig_mode = get_symmetric_quantization_config(is_qat=False)
    
    # methods to quantize individual layers/modules types are in quantizer
    quantizer = quantizer or TIDLRTQuantizer()
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
    
    # freezing the batchnorm update 
    for n in self.graph.nodes:
        # Args: input, weight, bias, running_mean, running_var, training, momentum, eps
        # We set the `training` flag to False here to freeze BN stats
        if n.target in [
            torch.ops.aten._native_batch_norm_legit.default,
            torch.ops.aten.cudnn_batch_norm.default,
        ]:
            new_args = list(n.args)
            new_args[5] = not(freeze_bn)
            n.args = new_args
    self.recompile()      
            
    return self


def unfreeze(self, freeze_bn=False, freeze_observers=False):
    freeze(self, freeze_bn, freeze_observers)
    return self


def forward(self, *input, **kwargs):
    return self(*input, **kwargs)


def convert(self, device="cpu", make_copy=False):
    if hasattr(self, '__quant_params__'):
        orig_quant_params = copy.deepcopy(self.__quant_params__)
    else:
        warnings.warn("__quant_params__ is missing in quant_func module. it may be due to a deepcopy.")
        orig_quant_params = None

    model = copy.deepcopy(self).eval() if make_copy else self.eval()
    model = model.to(device=device)
    model = convert_pt2e(model)
    torch.ao.quantization.move_exported_model_to_eval(model)
    model.eval = types.MethodType(train, model)

    if orig_quant_params:
        setattr(model, "__quant_params__", orig_quant_params)
    return model


def train(self, mode: bool = True):
    # hf transformers call train before every iteration, which messes with epochs tracked, will be needing a better logic for that #TODO
    # as of now, pass the expected number of epochs in num_batch_norm_update_epochs and num_observer_update_epochs variables, same is expected 
    # if the training code calls model.train() before every iteration 
    # put the model in expected mode
    if hasattr(self, "__train_backup__"):
        self.__train_backup__(mode=mode)
    # also freeze the params if required
    if mode is True:
        # torch.ao.quantization.move_exported_model_to_train(self)
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
        # torch.ao.quantization.move_exported_model_to_eval(self)
        freeze(self)
    #
    return self


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
    

def export(self, example_input, filename='model.onnx', opset_version=17, model_quant_format=None, preserve_qdq_model=True,
           simplify=True, skipped_optimizers=None, device='cpu', make_copy=True, is_converted=False, **export_kwargs):
    if not is_converted:
        model = convert(self, device=device, make_copy=make_copy)
    
# from: torch/ao/quantization/fx/graph_module.py
def _is_observed_module(module) -> bool:
    return hasattr(module, "meta") and "_observed_graph_module_attrs" in module.meta


def export(self, example_input, filename='model.onnx', opset_version=17, model_qconfig_format=None, preserve_qdq_model=True,
           simplify=True, skipped_optimizers=None, device='cpu', make_copy=True, is_converted=False, **export_kwargs):

    if _is_observed_module(self):
        model = convert(self, device=device, make_copy=make_copy)
    elif not is_converted:
        model = convert(self, device=device, make_copy=make_copy)
    else:
        model = self
        warnings.warn("model has already been converted before calling export. make sure it is done correctly.")

    # model, example_input = create_batch1_model(model, example_input)
    model = quant_utils.remove_loss_branch(model)
    quant_utils.register_onnx_symbolics()

    #from torch.fx import passes
    #g = passes.graph_drawer.FxGraphDrawer(model, "try_model")
    #with open('/home/a0491009/quantization/svg_files/prepared_qat_fx.svg', "wb") as f:
    #    f.write(g.get_dot_graph().create_svg())

    if model_qconfig_format == qconfig_types.QConfigFormat.INT_MODEL:
        # # Convert QDQ format to Int8 format
        import onnxruntime as ort
        qdq_filename = os.path.splitext(filename)[0] + '_qdq.onnx'
        torch.onnx.export(model, example_input.to('cpu'), qdq_filename, opset_version=opset_version, training=torch._C._onnx.TrainingMode.PRESERVE, **export_kwargs)
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
                example_inputs += tuple([val.to(device=device)])
        else:
            example_inputs = example_input.to(device=device)
        torch.onnx.export(model, example_inputs, filename, opset_version=opset_version, training=torch._C._onnx.TrainingMode.PRESERVE, **export_kwargs)

    if simplify:
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(filename)
        onnx_model, check = simplify(onnx_model, skipped_optimizers=skipped_optimizers)
        onnx.save(onnx_model, filename)
    
    
def create_batch1_model(orig_quantized_model, example_inputs):
    
    # # modifying the batch size of the input
    if isinstance(example_inputs, dict):
        for key, value in example_inputs.items():
            new_val = value[-1:]
            example_inputs[key] = new_val
    else:
        example_inputs = example_inputs[-1:]
    
    trained_state_dict = orig_quantized_model.state_dict()                        
    
    if hasattr(orig_quantized_model, "__quant_params__") and hasattr(orig_quantized_model.__quant_params__, 'original_model'):
        orig_model = orig_quantized_model.__quant_params__.original_model
        if isinstance(example_inputs, dict):
            m, guards = torchdynamo.export(orig_model, **example_inputs, aten_graph=True, assume_static_by_default=True)
            print("Dynamo export completed again")
        else:
            m, guards = torchdynamo.export(orig_model, example_inputs, aten_graph=True, assume_static_by_default=True)
    
        quantizer = orig_quantized_model.__quant_params__.quantizer
        model = prepare_pt2e(m, quantizer)
        if isinstance(example_inputs, dict):
            y = model(**example_inputs)
        else:
            y = model(example_inputs)
        model = convert_pt2e(model)
        
        model.load_state_dict(trained_state_dict)

    return model, example_inputs
