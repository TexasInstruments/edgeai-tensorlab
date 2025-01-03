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
import copy

from .. import constants
from ..import utils
from .basert_session import BaseRTSession


class TVMDLRSession(BaseRTSession):
    def __init__(self, session_name=constants.SESSION_NAME_TVMDLR, **kwargs):
        super().__init__(session_name=session_name, **kwargs)
        self.kwargs['input_data_layout'] = self.kwargs.get('input_data_layout', constants.NCHW)
        self.supported_machines = (constants.TARGET_MACHINE_PC_EMULATION, constants.TARGET_MACHINE_EVM)
        target_machine = self.kwargs['target_machine']
        assert target_machine in self.supported_machines, f'invalid target_machine {target_machine}'

    def import_model(self, calib_data, info_dict=None):
        # onnx and tvm are required only for model import
        # so import inside the function so that inference can be done without it
        from tvm import relay
        from tvm.relay.backend.contrib import tidl
        # prepare for actual model import
        super().import_model(calib_data, info_dict)

        self._get_input_output_details()

        model_file = self.kwargs['model_file']
        model_file0 = model_file[0] if isinstance(model_file, (list,tuple)) else model_file
        model_type = self.kwargs['model_type'] or os.path.splitext(model_file0)[1][1:]

        input_details = self.kwargs['input_details']
        input_shape = {inp_d['name']:inp_d['shape'] for inp_d in input_details}
        input_keys = list(input_shape.keys())

        if model_type == 'onnx':
            import onnx
            onnx_model = onnx.load_model(model_file0)
            tvm_model, params = relay.frontend.from_onnx(onnx_model, shape=input_shape)
        elif model_type == 'tflite':
            import tflite
            with open(model_file0, 'rb') as fp:
                tflite_model = tflite.Model.GetRootAsModel(fp.read(), 0)
            #
            tvm_model, params = relay.frontend.from_tflite(tflite_model, shape_dict=input_shape,
                                                   dtype_dict={k:'float32' for k in input_shape})
        elif model_type == 'mxnet':
            model_json, arg_params, aux_params = self._load_mxnet_model(model_file0)
            tvm_model, params = relay.frontend.from_mxnet(model_json, input_shape, arg_params=arg_params, aux_params=aux_params)
        else:
            assert False, f'unrecognized model type {model_type}'
        #

        calib_list = []
        for in_data in calib_data:
            in_data = utils.as_tuple(in_data)
            if self.input_normalizer is not None:
                in_data, _ = self.input_normalizer(in_data, {})
            #
            c_dict = {d_name:d for d_name, d in zip(input_keys,in_data)}
            calib_list.append(c_dict)
        #

        # Create the TIDL compiler with appropriate parameters
        compiler = tidl.TIDLCompiler(c7x_codegen=0, **self.kwargs['runtime_options'])

        artifacts_folder = self.kwargs['artifacts_folder']
        os.makedirs(artifacts_folder, exist_ok=True)

        # partition the graph into TIDL operations and TVM operations
        tvm_model, status = compiler.enable(tvm_model, params, calib_list)

        # the artifact files that are generated
        deploy_lib = 'deploy_lib.so'
        deploy_graph = 'deploy_graph.json'
        deploy_params = 'deploy_params.params'

        for target_machine in self.supported_machines:
            if target_machine == constants.TARGET_MACHINE_EVM:
                build_target = 'llvm -device=arm_cpu -mtriple=aarch64-linux-gnu'
                cross_cc_args = {'cc' : os.path.join(os.environ['ARM64_GCC_PATH'], 'bin', 'aarch64-none-linux-gnu-gcc')}
            elif target_machine == constants.TARGET_MACHINE_PC_EMULATION:
                build_target = 'llvm'
                cross_cc_args = {}
            else:
                assert False, f'unsupported target device {target_machine}'
            #

            # build the relay module into deployables
            with tidl.build_config(tidl_compiler=compiler):
                graph, lib, params = relay.build_module.build(tvm_model, target=build_target, params=params)

            # remove nodes / params not needed for inference
            tidl.remove_tidl_params(params)

            # save the deployables
            path_lib = os.path.join(artifacts_folder, f'{deploy_lib}.{target_machine}')
            path_graph = os.path.join(artifacts_folder, f'{deploy_graph}.{target_machine}')
            path_params = os.path.join(artifacts_folder, f'{deploy_params}.{target_machine}')

            lib.export_library(path_lib, **cross_cc_args)
            with open(path_graph, "w") as fo:
                fo.write(graph)
            #
            with open(path_params, "wb") as fo:
                fo.write(relay.save_param_dict(params))
            #
        #

        # create a symbolic link to the deploy_lib specified in target_machine
        os.chdir(artifacts_folder)
        target_machine = self.kwargs['target_machine']
        artifact_files = [deploy_lib, deploy_graph, deploy_params]
        for artifact_file in artifact_files:
            os.symlink(f'{artifact_file}.{target_machine}', artifact_file)
        #
        os.chdir(self.cwd)
        return info_dict

    def start_infer(self):
        super().start_infer()
        # create inference model
        self.interpreter = self._create_interpreter()
        self._get_input_output_details()
        os.chdir(self.cwd)
        return True

    def infer_frame(self, input, info_dict=None):
        super().infer_frame(input, info_dict)

        input_details = self.kwargs['input_details']
        input_shape = {inp_d['name']:inp_d['shape'] for inp_d in input_details}
        input_keys = list(input_shape.keys())

        in_data = utils.as_tuple(input)
        if self.input_normalizer is not None:
            in_data, _ = self.input_normalizer(in_data, {})
        #
        input_dict = {d_name:d for d_name, d in zip(input_keys,in_data)}
        # measure the time across only interpreter.run
        # time for setting the tensor and other overheads would be optimized out in c-api
        start_time = time.time()
        outputs = self.interpreter.run(input_dict)
        info_dict['session_invoke_time'] = (time.time() - start_time)
        self._update_output_details(outputs)
        return outputs, info_dict

    def set_runtime_option(self, option, value):
        advanced_options_prefix = 'advanced_options:'
        if advanced_options_prefix in option:
            option = option.replace(advanced_options_prefix, '')
            self.kwargs["runtime_options"]['advanced_options'][option] = value
        else:
            self.kwargs["runtime_options"][option] = value

    def get_runtime_option(self, option, default=None):
        advanced_options_prefix = 'advanced_options:'
        if advanced_options_prefix in option:
            option = option.replace(advanced_options_prefix, '')
            return self.kwargs["runtime_options"]['advanced_options'].get(option, default)
        else:
            return self.kwargs["runtime_options"].get(option, default)

    def _create_interpreter(self, is_import=False):
        # move the import inside the function, so that dlr needs to be installed
        # only if some one wants to use it
        from dlr import DLRModel
        artifacts_folder = self.kwargs['artifacts_folder']
        if not os.path.exists(artifacts_folder):
            return False
        #
        interpreter = DLRModel(artifacts_folder, 'cpu')
        return interpreter

    def _set_default_options(self):
        runtime_options = self.kwargs.get("runtime_options", {})
        default_options = {
            'platform':self.kwargs.get('platform', constants.TIDL_PLATFORM),
            'version':self.kwargs.get('version', constants.TIDL_VERSION),
            'data_layout':self.kwargs.get('data_layout', constants.NCHW),
            "tidl_tools_path": self.kwargs["tidl_tools_path"],
            "artifacts_folder": self.kwargs["artifacts_folder"],
            'tensor_bits':self.kwargs.get('tensor_bits', 8),
            # note: to add advanced options here, start it with 'advanced_options:'
            # example 'advanced_options:pre_batchnorm_fold':1
            # the code below will move those to a dict as required by the runtime interface
        }
        default_options.update(runtime_options)
        # tvm need advanced options as a dict
        # convert the entries starting with advanced_options: to a dict
        advanced_options_prefix = 'advanced_options:'
        advanced_options = {k.replace(advanced_options_prefix,''):v for k,v in default_options.items() \
                            if k.startswith(advanced_options_prefix)}
        default_options = {k:v for k,v in default_options.items() if not k.startswith(advanced_options_prefix)}
        default_options.update(dict(advanced_options=advanced_options))
        self.kwargs["runtime_options"] = default_options

    def _get_input_output_details(self):
        model_file = self.kwargs['model_file']
        model_file0 = model_file[0] if isinstance(model_file, (list,tuple)) else model_file
        model_type = self.kwargs['model_type'] or os.path.splitext(model_file0)[1][1:]
        if model_type == 'onnx':
            import onnxruntime
            sess_options = onnxruntime.SessionOptions()
            ep_list = ['CPUExecutionProvider']
            interpreter = onnxruntime.InferenceSession(model_file0, providers=ep_list,
                            provider_options=[{}], sess_options=sess_options)
            self._get_input_output_details_onnx(interpreter)
            del interpreter
        elif model_type == 'tflite':
            import tflite_runtime.interpreter as tflitert_interpreter
            runtime_options_temp = copy.deepcopy(self.kwargs["runtime_options"])
            runtime_options_temp['artifacts_folder'] = os.path.join(runtime_options_temp['artifacts_folder'], '_temp_details')
            self.kwargs["runtime_options"]["import"] = "yes"
            os.makedirs(runtime_options_temp['artifacts_folder'], exist_ok=True)
            self._clear_folder(runtime_options_temp['artifacts_folder'])
            interpreter = tflitert_interpreter.Interpreter(model_file0)
            self._get_input_output_details_tflite(interpreter)
            self._clear_folder(runtime_options_temp['artifacts_folder'], remove_base_folder=True)
            del interpreter
            del runtime_options_temp
        else:
            assert self.kwargs['input_details'] and self.kwargs['output_details'], \
                f'input_details and output_details must be provided for mxnet models'

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
