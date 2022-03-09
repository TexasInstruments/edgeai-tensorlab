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
import warnings
import copy
import struct
import numpy as np
import flatbuffers
import tflite_model
import tflite_runtime.interpreter as tflitert_interpreter
from .. import constants
from .. import utils
from .basert_session import BaseRTSession


class TFLiteRTSession(BaseRTSession):
    def __init__(self, session_name=constants.SESSION_NAME_TFLITERT, **kwargs):
        super().__init__(session_name=session_name, **kwargs)
        self.kwargs['input_data_layout'] = self.kwargs.get('input_data_layout', constants.NHWC)
        self.interpreter = None

    def import_model(self, calib_data, info_dict=None):
        super().import_model(calib_data)

        # create the underlying interpreter
        self.interpreter = self._create_interpreter(is_import=True)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        for c_data in calib_data:
            c_data = utils.as_tuple(c_data)
            if self.input_normalizer is not None:
                c_data, _ = self.input_normalizer(c_data, {})
            #
            for c_data_entry_idx, c_data_entry in enumerate(c_data):
                self._set_tensor(input_details[c_data_entry_idx], c_data_entry)
            #
            self.interpreter.invoke()
            outputs = [self._get_tensor(output_detail) for output_detail in output_details]
        #
        return info_dict

    def start_infer(self):
        super().start_infer()
        # now create the interpreter for inference
        self.interpreter = self._create_interpreter(is_import=False)
        os.chdir(self.cwd)
        return True

    def infer_frame(self, input, info_dict=None):
        super().infer_frame(input, info_dict)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        c_data = utils.as_tuple(input)
        if self.input_normalizer is not None:
            c_data, _ = self.input_normalizer(c_data, {})
        #
        for c_data_entry_idx, c_data_entry in enumerate(c_data):
            self._set_tensor(input_details[c_data_entry_idx], c_data_entry)
        #
        # measure the time across only interpreter.run
        # time for setting the tensor and other overheads would be optimized out in c-api
        start_time = time.time()
        self.interpreter.invoke()
        info_dict['session_invoke_time'] = (time.time() - start_time)
        outputs = [self._get_tensor(output_detail) for output_detail in output_details]
        return outputs, info_dict

    def set_runtime_option(self, option, value):
        self.kwargs["runtime_options"][option] = value

    def get_runtime_option(self, option, default=None):
        return self.kwargs["runtime_options"].get(option, default)

    def optimize_model(self):
        model_file = self.kwargs['model_file']
        input_mean = self.kwargs['input_mean']
        input_scale = self.kwargs['input_scale']
        out_model_path = model_file
        meanList = [x * -1 for x in input_mean]
        scaleList = input_scale
        modelBin = open(model_file, 'rb').read()
        if modelBin is None:
            print(f'Error: Could not open file {in_model_path}')
            return
        modelBin = bytearray(modelBin)
        model = tflite_model.Model.Model.GetRootAsModel(modelBin, 0)
        modelT = tflite_model.Model.ModelT.InitFromObj(model)

        #Add operators needed for preprocessing:
        self._set_tensor_properties(modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]], tflite_model.TensorType.TensorType.UINT8, 1.0, 0)
        mul_idx = self._add_new_operator(modelT, tflite_model.BuiltinOperator.BuiltinOperator.MUL)
        add_idx = self._add_new_operator(modelT, tflite_model.BuiltinOperator.BuiltinOperator.ADD)
        cast_idx = self._add_new_operator(modelT, tflite_model.BuiltinOperator.BuiltinOperator.CAST)

        in_cast_idx = self._add_new_operator(modelT, tflite_model.BuiltinOperator.BuiltinOperator.CAST)
        #Find argmax in the network:
        argMax_idx = self._get_argmax_idx(modelT)

        #Create a tensor for the "ADD" operator:
        bias_tensor = self._create_tensor(modelT, tflite_model.TensorType.TensorType.FLOAT32, None, [modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].shape[3]], bytearray(str("Preproc-bias"),'utf-8'))
        #Create a new buffer to store mean values:
        new_buffer = copy.copy(modelT.buffers[modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].buffer])
        new_buffer.data = struct.pack('%sf' % len(meanList), *meanList)
        modelT.buffers.append(new_buffer)
        new_buffer_idx = len(modelT.buffers) - 1
        bias_tensor.buffer = new_buffer_idx

        #Create a tensor for the "MUL" operator
        scale_tensor = copy.deepcopy(bias_tensor)
        scale_tensor.name  = bytearray(str("Preproc-scale"),'utf-8')
        #Create a new buffer to store the scale values:
        new_buffer = copy.copy(new_buffer)
        new_buffer.data = struct.pack('%sf' % len(scaleList), *scaleList)
        modelT.buffers.append(new_buffer)
        new_buffer_idx = len(modelT.buffers) - 1
        scale_tensor.buffer = new_buffer_idx

        #Append tensors into the tensor list:
        modelT.subgraphs[0].tensors.append(bias_tensor)
        bias_tensor_idx = len(modelT.subgraphs[0].tensors) - 1
        modelT.subgraphs[0].tensors.append(scale_tensor)
        scale_tensor_idx = len(modelT.subgraphs[0].tensors) - 1
        new_tensor = copy.deepcopy(modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].outputs[0]])
        new_tensor.name = bytearray((str(new_tensor.name, 'utf-8') + str("/Mul")),'utf-8')
        modelT.subgraphs[0].tensors.append(new_tensor)
        new_tensor_idx = len(modelT.subgraphs[0].tensors) - 1
        new_buffer = copy.deepcopy(modelT.buffers[modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].outputs[0]].buffer])
        modelT.buffers.append(new_buffer)
        new_buffer_idx = len(modelT.buffers) - 1
        modelT.subgraphs[0].tensors[new_tensor_idx].buffer = new_buffer_idx

        #Add the MUL Operator for scales:
        new_op = copy.deepcopy(modelT.subgraphs[0].operators[0])
        modelT.subgraphs[0].operators.insert(0,new_op)
        modelT.subgraphs[0].operators[0].outputs[0] = new_tensor_idx
        modelT.subgraphs[0].operators[0].inputs = [modelT.subgraphs[0].operators[1].inputs[0],scale_tensor_idx]
        modelT.subgraphs[0].operators[1].inputs[0] = new_tensor_idx
        modelT.subgraphs[0].tensors[new_tensor_idx].shape = modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].shape
        modelT.subgraphs[0].operators[0].opcodeIndex = mul_idx
        modelT.subgraphs[0].operators[0].builtinOptionsType = tflite_model.BuiltinOptions.BuiltinOptions.MulOptions
        modelT.subgraphs[0].operators[0].builtinOptions = tflite_model.MulOptions.MulOptionsT()

        #Add the ADD operator for mean:
        new_tensor = copy.deepcopy(modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].outputs[0]])
        new_tensor.name = bytearray((str(new_tensor.name, 'utf-8') + str("/Bias")),'utf-8')
        modelT.subgraphs[0].tensors.append(new_tensor)
        new_tensor_idx = len(modelT.subgraphs[0].tensors) - 1
        new_buffer = copy.deepcopy(modelT.buffers[modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].outputs[0]].buffer])
        modelT.buffers.append(new_buffer)
        new_buffer_idx = len(modelT.buffers) - 1
        modelT.subgraphs[0].tensors[new_tensor_idx].buffer = new_buffer_idx
        new_op_code = copy.deepcopy(modelT.operatorCodes[0])
        new_op = copy.deepcopy(modelT.subgraphs[0].operators[0])
        modelT.subgraphs[0].operators.insert(0,new_op)
        modelT.subgraphs[0].operators[0].outputs[0] = new_tensor_idx
        modelT.subgraphs[0].operators[0].inputs = [modelT.subgraphs[0].operators[1].inputs[0],bias_tensor_idx]
        modelT.subgraphs[0].operators[1].inputs[0] = new_tensor_idx
        modelT.subgraphs[0].tensors[new_tensor_idx].shape = modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].shape
        modelT.subgraphs[0].operators[0].opcodeIndex = add_idx
        modelT.subgraphs[0].operators[0].builtinOptionsType = tflite_model.BuiltinOptions.BuiltinOptions.AddOptions
        modelT.subgraphs[0].operators[0].builtinOptions = tflite_model.AddOptions.AddOptionsT()

        #Add the dequantize operator:
        new_tensor = copy.deepcopy(modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].outputs[0]])
        new_tensor.name = bytearray((str(new_tensor.name, 'utf-8') + str("/InCast")),'utf-8')
        modelT.subgraphs[0].tensors.append(new_tensor)
        new_tensor_idx = len(modelT.subgraphs[0].tensors) - 1
        new_buffer = copy.deepcopy(modelT.buffers[modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].outputs[0]].buffer])
        modelT.buffers.append(new_buffer)
        new_buffer_idx = len(modelT.buffers) - 1
        modelT.subgraphs[0].tensors[new_tensor_idx].buffer = new_buffer_idx
        new_op_code = copy.deepcopy(modelT.operatorCodes[0])
        new_op = copy.deepcopy(modelT.subgraphs[0].operators[0])
        modelT.subgraphs[0].operators.insert(0,new_op)
        modelT.subgraphs[0].operators[0].outputs[0] = new_tensor_idx
        modelT.subgraphs[0].operators[0].inputs = [modelT.subgraphs[0].operators[1].inputs[0]]
        modelT.subgraphs[0].operators[1].inputs[0] = new_tensor_idx
        modelT.subgraphs[0].tensors[new_tensor_idx].shape = modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]].shape
        modelT.subgraphs[0].operators[0].opcodeIndex = in_cast_idx
        modelT.subgraphs[0].operators[0].builtinOptionsType = tflite_model.BuiltinOptions.BuiltinOptions.CastOptions
        modelT.subgraphs[0].operators[0].builtinOptions = tflite_model.CastOptions.CastOptionsT()
        modelT.subgraphs[0].operators[0].builtinOptions.inDataType = tflite_model.TensorType.TensorType.UINT8
        modelT.subgraphs[0].tensors[new_tensor_idx].type  = tflite_model.TensorType.TensorType.FLOAT32

        #Detect and convert ArgMax's output data type:
        for operator in modelT.subgraphs[0].operators:
            #Find ARGMAX:
            if(operator.opcodeIndex == argMax_idx):
                if(modelT.subgraphs[0].tensors[operator.inputs[0]].shape[3] < 256): #Change dType only if #Classes can fit in UINT8
                    #Add CAST Op on ouput of Argmax:
                    new_op = copy.deepcopy(modelT.subgraphs[0].operators[0])
                    modelT.subgraphs[0].operators.append(new_op)
                    new_op_idx = len(modelT.subgraphs[0].operators) - 1

                    modelT.subgraphs[0].operators[new_op_idx].outputs[0] = operator.outputs[0]

                    new_tensor = copy.deepcopy(modelT.subgraphs[0].tensors[operator.outputs[0]])
                    new_tensor.name = bytearray((str(new_tensor.name, 'utf-8') + str("_org")),'utf-8')
                    modelT.subgraphs[0].tensors.append(new_tensor)
                    new_tensor_idx = len(modelT.subgraphs[0].tensors) - 1
                    new_buffer = copy.deepcopy(modelT.buffers[modelT.subgraphs[0].tensors[operator.outputs[0]].buffer])
                    modelT.buffers.append(new_buffer)
                    new_buffer_idx = len(modelT.buffers) - 1
                    modelT.subgraphs[0].tensors[new_tensor_idx].buffer = new_buffer_idx

                    operator.outputs[0] = new_tensor_idx

                    modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[new_op_idx].outputs[0]].type  = tflite_model.TensorType.TensorType.UINT8

                    modelT.subgraphs[0].operators[new_op_idx].inputs[0] = new_tensor_idx
                    modelT.subgraphs[0].operators[new_op_idx].opcodeIndex = cast_idx
                    modelT.subgraphs[0].operators[new_op_idx].builtinOptionsType = tflite_model.BuiltinOptions.BuiltinOptions.CastOptions
                    modelT.subgraphs[0].operators[new_op_idx].builtinOptions = tflite_model.CastOptions.CastOptionsT()
                    modelT.subgraphs[0].operators[new_op_idx].builtinOptions.outDataType = tflite_model.TensorType.TensorType.UINT8


        # Packs the object class into another flatbuffer.
        b2 = flatbuffers.Builder(0)
        b2.Finish(modelT.Pack(b2), b"TFL3")
        modelBuf = b2.Output()
        newFile = open(out_model_path, "wb")
        newFile.write(modelBuf)
        
        # set the mean and scale in kwarges to unity
        for m_idx, _ in enumerate(self.kwargs['input_mean']):
            self.kwargs['input_mean'] = 0.0
            self.kwargs['input_scale'] = 1.0
        #

    def _add_new_operator(self, modelT, operatorBuiltinCode):
        new_op_code                       = copy.deepcopy(modelT.operatorCodes[0])
        new_op_code.deprecatedBuiltinCode = operatorBuiltinCode
        modelT.operatorCodes.append(new_op_code)
        return (len(modelT.operatorCodes) - 1)

    def _get_argmax_idx(self, modelT):
        idx = 0
        for op in modelT.operatorCodes:
            if(op.deprecatedBuiltinCode == tflite_model.BuiltinOperator.BuiltinOperator.ARG_MAX):
                break
            idx = idx + 1
        return idx

    def _set_tensor_properties(self, tensor, dataType, scale, zeroPoint):
        tensor.type                   = dataType
        tensor.quantization.scale     = [scale]
        tensor.quantization.zeroPoint = [zeroPoint]

    def _create_tensor(self, modelT, dataType, quantization, tensorShape, tensorName):
        newTensor              = copy.deepcopy(modelT.subgraphs[0].tensors[modelT.subgraphs[0].operators[0].inputs[0]])
        newTensor.type         = dataType
        newTensor.quantization = quantization
        newTensor.shape        = tensorShape
        newTensor.name         = tensorName
        return newTensor

    def _create_interpreter(self, is_import):
        if self.kwargs['tidl_offload']:
            if is_import:
                self.kwargs["runtime_options"]["import"] = "yes"
                tidl_delegate = [tflitert_interpreter.load_delegate('tidl_model_import_tflite.so', self.kwargs["runtime_options"])]
            else:
                self.kwargs["runtime_options"]["import"] = "no"
                tidl_delegate = [tflitert_interpreter.load_delegate('libtidl_tfl_delegate.so', self.kwargs["runtime_options"])]
            #
            interpreter = tflitert_interpreter.Interpreter(model_path=self.kwargs['model_file'], experimental_delegates=tidl_delegate)
        else:
            interpreter = tflitert_interpreter.Interpreter(model_path=self.kwargs['model_file'])
        #
        interpreter.allocate_tensors()
        return interpreter

    def _set_default_options(self):
        runtime_options = self.kwargs.get("runtime_options", {})
        default_options = {
            "platform": constants.TIDL_PLATFORM,
            "version": constants.TIDL_VERSION_STR,
            "tidl_tools_path": self.kwargs["tidl_tools_path"],
            "artifacts_folder": self.kwargs["artifacts_folder"],
            "tensor_bits": self.kwargs.get("tensor_bits", 8),
            "import": self.kwargs.get("import", 'no'),
            # note: to add advanced options here, start it with 'advanced_options:'
            # example 'advanced_options:pre_batchnorm_fold':1
        }
        default_options.update(runtime_options)
        self.kwargs["runtime_options"] = default_options

    def _get_input_shape_tflite(self):
        input_shape = {}
        model_input_details = self.interpreter.get_input_details()
        for model_input in model_input_details:
            name = model_input['name']
            shape = model_input['shape']
            input_shape.update({name:shape})
        #
        return input_shape

    def _set_tensor(self, model_input, tensor):
        if model_input['dtype'] == np.int8:
            # scale, zero_point = model_input['quantization']
            # tensor = np.clip(np.round(tensor/scale + zero_point), -128, 127)
            tensor = np.array(tensor, dtype=np.int8)
        elif model_input['dtype'] == np.uint8:
            # scale, zero_point = model_input['quantization']
            # tensor = np.clip(np.round(tensor/scale + zero_point), 0, 255)
            tensor = np.array(tensor, dtype=np.uint8)
        #
        self.interpreter.set_tensor(model_input['index'], tensor)

    def _get_tensor(self, model_output):
        tensor = self.interpreter.get_tensor(model_output['index'])
        if model_output['dtype'] == np.int8 or model_output['dtype']  == np.uint8:
            scale, zero_point = model_output['quantization']
            tensor = np.array(tensor, dtype=np.float32)
            tensor = (tensor - zero_point) / scale
        #
        return tensor
