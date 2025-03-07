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
import copy

from . import presets
from .basert_runtime import BaseRuntimeWrapper

tvmdlr = "dlr"

class TVMDLRRuntimeWrapper(BaseRuntimeWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_import_done = False
        self._start_inference_done = False
        self._num_run_import = 0
        self._input_list = []
        self.supported_machines = (
            presets.TARGET_MACHINE_PC_EMULATION,
            presets.TARGET_MACHINE_EVM
        )

    def start_import(self):
        self.is_import = True
        self._calibration_frames = self.kwargs["runtime_options"]["advanced_options:calibration_frames"]
        self.kwargs["runtime_options"] = self._set_default_options(self.kwargs["runtime_options"])
       # tvm/dlr requires input shape in prepare_for_import - so moved this ahead
        self.kwargs['input_details'] = self.get_input_details(None, self.kwargs.get('input_details', None))
        self.kwargs['output_details'] = self.get_output_details(None, self.kwargs.get('output_details', None))
        self._input_list = []
        self._start_import_done = True
        return True

    def run_import(self, input_data, output_keys=None):
        if not self._start_import_done:
            self.start_import()
        #
        self._num_run_import += 1

        #_format_input_data was not called yet, as shapes were not available - call it here:
        input_data = self._format_input_data(input_data)
        self._input_list.append(input_data)

        if len(self._input_list) == self._calibration_frames:
            self.interpreter = self._create_interpreter_for_import(self._input_list)
        elif len(self._input_list) > self._calibration_frames:
            print(f"WARNING: not need to call run_import more than calibration_frames = {self._calibration_frames}")
        #
        return self.interpreter

    def start_inference(self):
        self.is_import = False
        self._calibration_frames = self.kwargs["runtime_options"]["advanced_options:calibration_frames"]
        self.kwargs["runtime_options"] = self._set_default_options(self.kwargs["runtime_options"])
        self.kwargs['input_details'] = self.get_input_details(None, self.kwargs.get('input_details', None))
        self.kwargs['output_details'] = self.get_output_details(None, self.kwargs.get('output_details', None))
        # moved the import inside the function, so that dlr needs to be installed only if someone wants to use it
        
        artifacts_folder = self.kwargs['artifacts_folder']
        if tvmdlr=="dlr":
            from dlr import DLRModel
            if not os.path.exists(artifacts_folder):
                return False
            #
            self.interpreter = DLRModel(artifacts_folder, 'cpu')
        else:
            import tvm
            from tvm.contrib import graph_executor as runtime

            loaded_json = open(artifacts_folder + "/deploy_graph.json").read()
            loaded_lib = tvm.runtime.load_module(artifacts_folder + "/deploy_lib.so","so")
            loaded_params = bytearray(open(artifacts_folder + "/deploy_params.params", "rb").read())

            # create a runtime executor module
            self.interpreter = runtime.create(loaded_json, loaded_lib, tvm.cpu())
            self.interpreter.load_params(loaded_params)
        self._start_inference_done = True
        return self.interpreter

    def run_inference(self, input_data, output_keys=None):
        if not self._start_inference_done:
            self.start_inference()
        #
        input_data = self._format_input_data(input_data)
        return self._run(input_data, output_keys)

    def _run(self, input_data, output_keys=None):
        # if model needs additional inputs given in extra_inputs
        if self.kwargs.get('extra_inputs'):
            input_data.update(self.kwargs['extra_inputs'])
        #

        if tvmdlr == "dlr":
            outputs = self.interpreter.run(input_data)
        else:
            for key, value in input_data.items():
                self.interpreter.set_input(key, value)
            
            self.interpreter.run()

            outputs = []
            for i in range(self.interpreter.get_num_outputs()):
                outputs.append(self.interpreter.get_output(i).asnumpy())

        return outputs

    def _create_interpreter_for_import(self, calib_list):
        # onnx and tvm are required only for model import
        # so import inside the function so that inference can be done without it
        from tvm import relay
        from tvm.relay.backend.contrib import tidl

        model_file = self.kwargs['model_file']
        model_file0 = model_file[0] if isinstance(model_file, (list,tuple)) else model_file
        model_type = self.kwargs.get('model_type',None) or os.path.splitext(model_file0)[1][1:]

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

       # Create the TIDL compiler with appropriate parameters
        if (not self.kwargs.get('tidl_offload', 1)):
            self.kwargs['runtime_options']['max_num_subgraphs'] = 0
        #
        compiler = tidl.TIDLCompiler(
            c7x_codegen=self.kwargs.get('c7x_codegen', 0),
            **self.kwargs['runtime_options'],
        )

        artifacts_folder = self.kwargs['artifacts_folder']
        os.makedirs(artifacts_folder, exist_ok=True)

        # partition the graph into TIDL operations and TVM operations
        tvm_model, status = compiler.enable(tvm_model, params, calib_list)

        # the artifact files that are generated
        deploy_lib = 'deploy_lib.so'
        deploy_graph = 'deploy_graph.json'
        deploy_params = 'deploy_params.params'

        for target_machine in self.supported_machines:
            if target_machine == presets.TARGET_MACHINE_EVM:
                build_target = 'llvm -device=arm_cpu -mtriple=aarch64-linux-gnu'
                cross_cc_args = {'cc' : os.path.join(os.environ['ARM64_GCC_PATH'], 'bin', 'aarch64-none-linux-gnu-gcc')}
            elif target_machine == presets.TARGET_MACHINE_PC_EMULATION:
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
        artifacts_folder = self.kwargs['artifacts_folder']
        target_machine = self.kwargs.get('target_machine', presets.TARGET_MACHINE_PC_EMULATION)
        cwd = os.getcwd()
        os.chdir(artifacts_folder)
        artifact_files = [deploy_lib, deploy_graph, deploy_params]
        for artifact_file in artifact_files:
            os.symlink(f'{artifact_file}.{target_machine}', artifact_file)
        #
        os.chdir(cwd)
        return status

    def _format_input_data(self, input_data):
        if isinstance(input_data, dict):
            return input_data

        if not isinstance(input_data, (list,tuple)):
            input_data = (input_data,)

        input_details = self.kwargs['input_details']
        input_shape = {inp_d['name']:inp_d['shape'] for inp_d in input_details}
        input_keys = list(input_shape.keys())
        input_data = {d_name:d for d_name, d in zip(input_keys,input_data)}
        return input_data

    def _set_default_options(self, runtime_options):
        default_options = {
            'platform':self.kwargs.get('platform', presets.TIDL_PLATFORM),
            'version':self.kwargs.get('version', presets.TIDL_VERSION),
            'data_layout':self.kwargs.get('data_layout', presets.NCHW),
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
        return default_options

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

    def get_input_details(self, dlr_interpreter, input_details=None):
        if input_details is None:
            model_file = self.kwargs['model_file']
            model_file0 = model_file[0] if isinstance(model_file, (list,tuple)) else model_file
            model_type = self.kwargs.get('model_type',None) or os.path.splitext(model_file0)[1][1:]
            if model_type == 'onnx':
                import onnxruntime
                sess_options = onnxruntime.SessionOptions()
                ep_list = ['CPUExecutionProvider']
                interpreter = onnxruntime.InferenceSession(model_file0, providers=ep_list,
                                provider_options=[{}], sess_options=sess_options)
                input_details = super()._get_input_details_onnx(interpreter, input_details)
                del interpreter
            elif model_type == 'tflite':
                import tflite_runtime.interpreter as tflitert_interpreter
                runtime_options_temp = copy.deepcopy(self.kwargs["runtime_options"])
                runtime_options_temp['artifacts_folder'] = os.path.join(runtime_options_temp['artifacts_folder'], '_temp_details')
                self.kwargs["runtime_options"]["import"] = "yes"
                os.makedirs(runtime_options_temp['artifacts_folder'], exist_ok=True)
                self._clear_folder(runtime_options_temp['artifacts_folder'])
                interpreter = tflitert_interpreter.Interpreter(model_file0)
                input_details = self._get_input_details_tflite(interpreter, input_details)
                self._clear_folder(runtime_options_temp['artifacts_folder'], remove_base_folder=True)
                del interpreter
                del runtime_options_temp
            else:
                raise RuntimeError('input_details can be obtained for onnx and tiflite models - for others, it must be provided')
            #
        #
        return input_details

    def get_output_details(self, dlr_interpreter, output_details=None):
        if output_details is None:
            model_file = self.kwargs['model_file']
            model_file0 = model_file[0] if isinstance(model_file, (list,tuple)) else model_file
            model_type = self.kwargs.get('model_type',None) or os.path.splitext(model_file0)[1][1:]
            if model_type == 'onnx':
                import onnxruntime
                sess_options = onnxruntime.SessionOptions()
                ep_list = ['CPUExecutionProvider']
                interpreter = onnxruntime.InferenceSession(model_file0, providers=ep_list,
                                provider_options=[{}], sess_options=sess_options)
                output_details = super()._get_output_details_onnx(interpreter, output_details)
                del interpreter
            elif model_type == 'tflite':
                import tflite_runtime.interpreter as tflitert_interpreter
                runtime_options_temp = copy.deepcopy(self.kwargs["runtime_options"])
                runtime_options_temp['artifacts_folder'] = os.path.join(runtime_options_temp['artifacts_folder'], '_temp_details')
                self.kwargs["runtime_options"]["import"] = "yes"
                os.makedirs(runtime_options_temp['artifacts_folder'], exist_ok=True)
                self._clear_folder(runtime_options_temp['artifacts_folder'])
                interpreter = tflitert_interpreter.Interpreter(model_file0)
                output_details = self._get_output_details_tflite(interpreter, output_details)
                self._clear_folder(runtime_options_temp['artifacts_folder'], remove_base_folder=True)
                del interpreter
                del runtime_options_temp
            else:
                raise RuntimeError('output_details can be obtained for onnx and tiflite models - for others, it must be provided')
            #
        #
        return output_details

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
