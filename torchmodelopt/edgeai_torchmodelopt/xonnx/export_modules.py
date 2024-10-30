#################################################################################
# Copyright (c) 2018-2024, Texas Instruments Incorporated - http://www.ti.com
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
import os.path
import types

import torch
import torchvision
import onnx
import onnxscript # needed for dynamo export
import onnxsim


from edgeai_torchmodelopt.xmodelopt.utils.hooks import add_example_args_kwargs

# in the named modules, the whole model is stored with the key empty string
# so if the whole model is to be exported, use this key
WHOLE_MODEL_KEY_IN_NAMED_MODULES = ''


def _add_forward_with_kwargs(module, *args, **kwargs):
    """
    make a wrapper to store the args and kwargs inside and use it during the forward call
    """
    module.__forward_backup = module.forward
    __forward_func_new = lambda *module_args, **module_kwargs: module.__forward_backup(*args, **kwargs)
    module.forward = types.MethodType(__forward_func_new, module)


def _remove_forward_with_kwargs(module):
    module.forward = module.__forward_backup
    del module.__forward_backup


def _export_named_module(named_modules, module_name, exported_filename, opset_version, dynamo):
        module = named_modules[module_name]
        module_args = tuple(module._example_inputs) if isinstance(module._example_inputs, (list,tuple)) \
                        else (module._example_inputs,)
        module_kwargs = module._example_kwargs

        # torch.onnx.export accepts only args, it doesn't accept kwargs
        # make a wrapper to store the kwargs inside and use it during the forward call
        _add_forward_with_kwargs(module, *module_args, **module_kwargs)

        # test if the forward works
        module_outputs = module(*module_args, **module_kwargs)

        # onnx export
        if dynamo:
            # export_options = torch.onnx.ExportOptions(dynamic_shapes=False)
            # onnx_program = torch.onnx.dynamo_export(module, module_inputs, export_options=export_options)
            # onnx_program.save(exported_filename)
            torch.onnx.export(module, module_args, exported_filename, dynamo=dynamo)
        else:
            torch.onnx.export(module, module_args, exported_filename, opset_version=opset_version)

        _remove_forward_with_kwargs(module)

        # simplify
        onnx_model = onnx.load(exported_filename)
        simplified_model, model_ok = onnxsim.simplify(onnx_model)
        onnx.save(simplified_model, exported_filename)


def export_modules(model, example_input, filename, module_names, opset_version, dynamo=False):
    """
    Export a model into onnx files in parts

    The parts are specified by their names specified as a list of module_names.
    dict(model.named_modules()) returns a lists of all names as submodules in a model
    list(dict(model.named_modules()).keys()) returns a list of all submodule keys
    The modules names specified can be a subset of those names.
    """

    model.eval()

    example_inputs = tuple(example_input) if isinstance(example_input, (list,tuple)) else (example_input,)

    if module_names is None:
        named_childern = dict(model.named_children())
        module_names = list(named_childern.keys())

    transformation_dict = {m:None for m in module_names}
    add_example_args_kwargs(model, example_inputs, transformation_dict=transformation_dict)

    filename_base = os.path.splitext(os.path.basename(filename))[0]

    named_modules = dict(model.named_modules())
    for module_name in transformation_dict:
        exported_filename = f"{filename_base}_{module_name}.onnx"
        try:
            print(f"Exporting - {module_name}")
            _export_named_module(named_modules, module_name, exported_filename, opset_version=opset_version, dynamo=dynamo)
            print(f"Export - {module_name}: COMPLETED")
        except Exception as e:
            print(f"Export - {module_name}: FAILED, {e}")


if __name__ == "__main__":
    model = torchvision.models.detection.maskrcnn_resnet50_fpn()
    example_input = torch.rand([1,3,640,640])
    module_names = [WHOLE_MODEL_KEY_IN_NAMED_MODULES, 'backbone', 'rpn', 'transform', 'roi_heads']
    export_modules(model, example_input, "model.onnx", module_names, opset_version=17)