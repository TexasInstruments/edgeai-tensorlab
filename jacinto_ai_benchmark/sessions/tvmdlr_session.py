# Copyright (c) 2018-2021, Texas Instruments
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

import os
import time
from dlr import DLRModel
from .. import constants
from .basert_session import BaseRTSession
from ..import utils


class TVMDLRSession(BaseRTSession):
    def __init__(self, session_name=constants.SESSION_NAME_TVMDLR, **kwargs):
        super().__init__(session_name=session_name, **kwargs)
        self.interpreter = None

    def import_model(self, calib_data, info_dict=None):
        # onnx and tvm are required only for model import
        # so import inside the function so that inference can be done without it
        from tvm import relay
        from tvm.relay.backend.contrib import tidl
        # prepare for actual model import
        super().import_model(calib_data, info_dict)

        # this chdir() is required for the import to work.
        interpreter_folder = os.path.join(os.environ['TIDL_BASE_PATH'], 'ti_dl/test/tvm-dlr')
        os.chdir(interpreter_folder)

        model_path = self.kwargs['model_path']
        model_path0 = model_path[0] if isinstance(model_path, (list,tuple)) else model_path
        model_type = self.kwargs['model_type'] or os.path.splitext(model_path0)[1][1:]
        if model_type == 'mxnet':
            model_json, arg_params, aux_params = self._load_mxnet_model(model_path)
            assert self.kwargs['input_shape'] is not None, 'input_shape must be given'
            input_shape = self.kwargs['input_shape']
            input_keys = list(input_shape.keys())
            tvm_model, params = relay.frontend.from_mxnet(model_json, input_shape, arg_params=arg_params, aux_params=aux_params)
        elif model_type == 'onnx':
            import onnx
            onnx_model = onnx.load(model_path)
            if self.kwargs['input_shape'] is None:
                self.kwargs['input_shape'] = self._get_input_shape_onnx(onnx_model)
            #
            input_shape = self.kwargs['input_shape']
            input_keys = list(input_shape.keys())
            tvm_model, params = relay.frontend.from_onnx(onnx_model, shape=input_shape)
        elif model_type == 'tflite':
            import tflite
            if self.kwargs['input_shape'] is None:
                self.kwargs['input_shape'] = self._get_input_shape_tflite()
            #
            input_shape = self.kwargs['input_shape']
            input_keys = list(input_shape.keys())
            with open(model_path, 'rb') as fp:
                tflite_model = tflite.Model.GetRootAsModel(fp.read(), 0)
            #
            tvm_model, params = relay.frontend.from_tflite(tflite_model, shape_dict=input_shape,
                                                   dtype_dict={k:'float32' for k in input_shape})
        else:
            assert False, f'unrecognized model type {model_type}'
        #

        calib_list = []
        for c_data in calib_data:
            c_data = utils.as_tuple(c_data)
            c_dict = {d_name:d for d_name, d in zip(input_keys,c_data)}
            calib_list.append(c_dict)
        #

        # Create the TIDL compiler with appropriate parameters
        compiler = tidl.TIDLCompiler(**self.kwargs['compiler_options'])

        supported_devices = self.kwargs['supported_devices'] if (self.kwargs['supported_devices'] is not None) \
            else (self.kwargs['target_device'],)

        for target_device in supported_devices:
            if target_device == 'j7':
                build_target = 'llvm -device=arm_cpu -mtriple=aarch64-linux-gnu'
                cross_cc_args = {'cc' : os.path.join(os.environ['ARM64_GCC_PATH'], 'bin', 'aarch64-none-linux-gnu-gcc')}
            elif target_device == 'pc':
                build_target = 'llvm'
                cross_cc_args = {}
            else:
                assert False, f'unsupported target device {target_device}'
            #

            generate_artifacts_folder = self._get_target_artifacts_folder(target_device)
            os.makedirs(generate_artifacts_folder, exist_ok=True)

            # partition the graph into TIDL operations and TVM operations
            tvm_model, status = compiler.enable(tvm_model, params, calib_list)

            # build the relay module into deployables
            with tidl.build_config(tidl_compiler=compiler):
                graph, lib, params = relay.build_module.build(tvm_model, target=build_target, params=params)

            # remove nodes / params not needed for inference
            tidl.remove_tidl_params(params)

            # save the deployables
            path_lib = os.path.join(generate_artifacts_folder, 'deploy_lib.so')
            path_graph = os.path.join(generate_artifacts_folder, 'deploy_graph.json')
            path_params = os.path.join(generate_artifacts_folder, 'deploy_params.params')
            lib.export_library(path_lib, **cross_cc_args)
            with open(path_graph, "w") as fo:
                fo.write(graph)
            #
            with open(path_params, "wb") as fo:
                fo.write(relay.save_param_dict(params))
            #
        #
        os.chdir(self.cwd)
        return info_dict

    def start_infer(self):
        super().start_infer()
        # create inference model
        self.interpreter = self._create_interpreter()
        if self.kwargs['input_shape'] is None:
            # get the input names from DLR model
            # don't know how to get the input shape from dlr model, but that's not requried.
            input_names = self.interpreter.get_input_names()
            input_names = utils.as_list_or_tuple(input_names)
            self.kwargs['input_shape'] = {n:None for n in input_names}
        #
        os.chdir(self.cwd)
        return True

    def infer_frame(self, input, info_dict=None):
        super().infer_frame(input, info_dict)
        input_shape = self.kwargs['input_shape']
        input_keys = list(input_shape.keys())
        in_data = utils.as_tuple(input)
        input_dict = {d_name:d for d_name, d in zip(input_keys,in_data)}
        # measure the time across only interpreter.run
        # time for setting the tensor and other overheads would be optimized out in c-api
        start_time = time.time()
        output = self.interpreter.run(input_dict)
        info_dict['session_invoke_time'] = (time.time() - start_time)
        return output, info_dict

    def _create_interpreter(self, is_import=False):
        target_artifacts_folder = self._get_target_artifacts_folder(self.kwargs['target_device'])
        if not os.path.exists(target_artifacts_folder):
            return False
        #
        interpreter = DLRModel(target_artifacts_folder, 'cpu')
        return interpreter

    def _set_default_options(self):
        default_options = {
            'platform':self.kwargs.get('platform', 'J7'),
            'version':self.kwargs.get('version', (7,0)),
            'data_layout':self.kwargs.get('data_layout', constants.NCHW),
            'artifacts_folder':self.kwargs.get('artifacts_folder', None),
            'tidl_tools_path':os.path.join(os.environ['TIDL_BASE_PATH'], 'tidl_tools'),
            'tidl_tensor_bits':self.kwargs.get('tidl_tensor_bits', 8),
            'num_tidl_subgraphs':self.kwargs.get('num_tidl_subgraphs', 16),
            'debug_level':self.kwargs.get('debug_level', 0),
            'power_of_2_quantization':self.kwargs.get('power_of_2_quantization', 'off'),
            'pre_batchnorm_fold':self.kwargs.get('pre_batchnorm_fold', 1),
            'enable_high_resolution_optimization':self.kwargs.get('enable_high_resolution_optimization', 'off'),
            'tidl_calibration_accuracy_level':self.kwargs.get('tidl_calibration_accuracy_level', 1),
            'tidl_calibration_options':self.kwargs.get('tidl_calibration_options', {}),
            'reserved_compile_constraints_flag':self.kwargs.get('reserved_compile_constraints_flag', None)
        }
        compiler_options = self.kwargs.get("compiler_options", {})
        compiler_options = utils.dict_update_conditional(compiler_options, default_options, inplace=True)
        self.kwargs['compiler_options'] = compiler_options

    def _get_input_shape_onnx(self, onnx_model):
        input_shape = {}
        num_inputs = self.kwargs['num_inputs']
        for input_idx in range(num_inputs):
            input_i = onnx_model.graph.input[input_idx]
            name = input_i.name
            shape = [dim.dim_value for dim in input_i.type.tensor_type.shape.dim]
            input_shape.update({name:shape})
        #
        return input_shape

    def _get_input_shape_tflite(self):
        import tflite_runtime.interpreter as tflitert_interpreter
        interpreter = tflitert_interpreter.Interpreter(model_path=self.kwargs['model_path'])
        input_shape = {}
        model_input_details = interpreter.get_input_details()
        for model_input in model_input_details:
            name = model_input['name']
            shape = list(model_input['shape'])
            input_shape.update({name:shape})
        #
        del interpreter
        return input_shape

    def _get_target_artifacts_folder(self, target_device):
        target_artifacts_folder = self.kwargs['artifacts_folder']
        # we need to create multiple artifacts folder only if supported devices is specified.
        if self.kwargs['supported_devices'] is not None:
            target_artifacts_folder = os.path.join(target_artifacts_folder, target_device)
        #
        return target_artifacts_folder

    def _load_mxnet_model(self, model_path):
        import mxnet
        assert isinstance(model_path, list) and len(model_path) == 2, 'mxnet model path must be a list of size 2'

        model_json = mxnet.symbol.load(model_path[0])
        save_dict = mxnet.ndarray.load(model_path[1])
        arg_params = {}
        aux_params = {}
        for key, param in save_dict.items():
            tp, name = key.split(':', 1)
            if tp == 'arg':
                arg_params[name] = param
            elif tp == 'aux':
                aux_params[name] = param
            #
        #
        return model_json, arg_params, aux_params


if __name__ == '__main__':
    tvm_model = TVMDLRSession()
