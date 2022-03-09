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
import numpy as np
import warnings
import onnx
import onnxruntime
from .. import utils
from .. import constants
from .basert_session import BaseRTSession


class ONNXRTSession(BaseRTSession):
    def __init__(self, session_name=constants.SESSION_NAME_ONNXRT, **kwargs):
        super().__init__(session_name=session_name, **kwargs)
        self.kwargs['input_data_layout'] = self.kwargs.get('input_data_layout', constants.NCHW)
        self.interpreter = None

    def start(self):
        super().start()

    def import_model(self, calib_data, info_dict=None):
        super().import_model(calib_data)

        # create the underlying interpreter
        self.interpreter = self._create_interpreter(is_import=True)
        # check if the shape of data being provided matches with what the model expects
        if self.kwargs['input_shape'] is None:
            self.kwargs['input_shape'] = self._get_input_shape_onnxrt()
        #
        # provide the calibration data and run the import
        for c_data in calib_data:
            input_keys = list(self.kwargs['input_shape'].keys())
            c_data = utils.as_tuple(c_data)
            if self.input_normalizer is not None:
                c_data, _ = self.input_normalizer(c_data, {})
            #
            calib_dict = {d_name:d for d_name, d in zip(input_keys,c_data)}
            # model may need additional inputs given in extra_inputs
            if self.kwargs['extra_inputs'] is not None:
                calib_dict.update(self.kwargs['extra_inputs'])
            #
            output_keys = list(self.kwargs['output_shape'].keys()) \
                if self.kwargs['output_shape'] is not None else None
            # run the actual import step
            outputs = self.interpreter.run(output_keys, calib_dict)
        #
        return info_dict

    def start_infer(self):
        super().start_infer()
        # create the underlying interpreter
        self.interpreter = self._create_interpreter(is_import=False)
        # input_shape is needed during inference - get it if it is not given
        if self.kwargs['input_shape'] is None:
            self.kwargs['input_shape'] = self._get_input_shape_onnxrt()
        #
        os.chdir(self.cwd)
        return True

    def infer_frame(self, input, info_dict=None):
        super().infer_frame(input, info_dict)
        input_keys = list(self.kwargs['input_shape'].keys())
        in_data = utils.as_tuple(input)
        if self.input_normalizer is not None:
            c_data, _ = self.input_normalizer(c_data, {})
        #
        input_dict = {d_name:d for d_name, d in zip(input_keys,in_data)}
        # model needs additional inputs given in extra_inputs
        if self.kwargs['extra_inputs'] is not None:
            input_dict.update(self.kwargs['extra_inputs'])
        #
        # output_shape is not mandatory, output_keys can be None
        output_keys = list(self.kwargs['output_shape'].keys()) \
            if self.kwargs['output_shape'] is not None else None
        # run the actual inference
        start_time = time.time()
        outputs = self.interpreter.run(output_keys, input_dict)
        info_dict['session_invoke_time'] = (time.time() - start_time)
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
        #Read Model
        meanList = [x * -1 for x in input_mean]
        scaleList = input_scale
        model = onnx.load_model(model_file)
        op = onnx.OperatorSetIdProto()
        #Track orginal opset:
        op.version = model.opset_import[0].version
        #Get Graph:
        originalGraph = model.graph
        #Get Nodes:
        originalNodes = originalGraph.node
        #Get Initializers:
        originalInitializers = originalGraph.initializer
        #Create Lists
        nodeList = [node for node in originalNodes]
        initList = [init for init in originalInitializers]

        nInCh = int(originalGraph.input[0].type.tensor_type.shape.dim[1].dim_value)

        #Input & Output Dimensions:
        inDims = tuple([x.dim_value for x in originalGraph.input[0].type.tensor_type.shape.dim])
        outDims = tuple([x.dim_value for x in originalGraph.output[0].type.tensor_type.shape.dim])

        #Construct bias & scale tensors
        biasTensor = onnx.helper.make_tensor("TIDL_preProc_Bias",onnx.TensorProto.FLOAT,[1,nInCh, 1, 1],np.array(meanList,dtype=np.float32))
        scaleTensor = onnx.helper.make_tensor("TIDL_preProc_Scale",onnx.TensorProto.FLOAT,[1, nInCh, 1, 1],np.array(scaleList,dtype=np.float32))

        #Add these tensors to initList:
        initList.append(biasTensor)
        initList.append(scaleTensor)

        #Cast Node:
        attrib_dict = {"to":onnx.TensorProto.FLOAT}
        cast = onnx.helper.make_node('Cast',inputs=[originalGraph.input[0].name+"Net_IN"],outputs=['TIDL_cast_in'], **attrib_dict)

        #Add Node:
        addNode = onnx.helper.make_node('Add',inputs=["TIDL_cast_in","TIDL_preProc_Bias"],outputs=["TIDL_Scale_In"])

        #Scale Node:
        scaleNode = onnx.helper.make_node('Mul',inputs=["TIDL_Scale_In","TIDL_preProc_Scale"],outputs=[originalGraph.input[0].name]) #Assumption that input[0].name is the input node

        nodeList = [cast, addNode, scaleNode] + nodeList #Toplogically Sorted

        outSequence = originalGraph.output
        #Check for Argmax:
        for node in nodeList:
            if node.op_type == "ArgMax":
                #Check if it is final output:
                if node.output[0] == originalGraph.output[0].name:
                    #Argmax Output is final output:
                    attrib_dict_1 = {"to":TensorProto.UINT8}
                    cast_out = onnx.helper.make_node('Cast',inputs=[originalGraph.output[0].name],outputs=[originalGraph.output[0].name+'TIDL_cast_out'], **attrib_dict_1)
                    nodeList = nodeList + [cast_out] #Toplogically Sorted
                    outSequence = [helper.make_tensor_value_info(originalGraph.output[0].name+'TIDL_cast_out', TensorProto.UINT8,outDims)]

        #Construct Graph:
        newGraph = onnx.helper.make_graph(
            nodeList,
            'Rev_Model',
            [onnx.helper.make_tensor_value_info(originalGraph.input[0].name+"Net_IN", onnx.TensorProto.UINT8, inDims)],
            outSequence,
            initList
            )
        #Construct Model:
        op.version = 11
        model_def_noShape = onnx.helper.make_model(newGraph, producer_name='onnx-TIDL', opset_imports=[op])
        model_def = onnx.shape_inference.infer_shapes(model_def_noShape)

        try:
            onnx.checker.check_model(model_def)
        except onnx.checker.ValidationError as e:
            print('Converted model is invalid: %s' % e)
        else:
            print('Converted model is valid!')
            onnx.save_model(model_def, out_model_path)
        #
        
        # set the mean and scale in kwarges to unity
        for m_idx, _ in enumerate(self.kwargs['input_mean']):
            self.kwargs['input_mean'] = 0.0
            self.kwargs['input_scale'] = 1.0
        #

    def _create_interpreter(self, is_import):
        # pass options to pybind
        if is_import:
            self.kwargs["runtime_options"]["import"] = "yes"
        else:
            self.kwargs["runtime_options"]["import"] = "no"
        #
        runtime_options = self.kwargs["runtime_options"]
        sess_options = onnxruntime.SessionOptions()

        if self.kwargs['tidl_offload']:
            ep_list = ['TIDLCompilationProvider', 'CPUExecutionProvider'] if is_import else \
                      ['TIDLExecutionProvider', 'CPUExecutionProvider']
            interpreter = onnxruntime.InferenceSession(self.kwargs['model_file'], providers=ep_list,
                            provider_options=[runtime_options, {}], sess_options=sess_options)
        else:
            ep_list = ['CPUExecutionProvider']
            interpreter = onnxruntime.InferenceSession(self.kwargs['model_file'], providers=ep_list,
                            provider_options=[{}], sess_options=sess_options)
        #
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

    def _get_input_shape_onnxrt(self):
        input_details = self.interpreter.get_inputs()
        input_shape = {}
        for inp in input_details:
            input_shape.update({inp.name:inp.shape})
        #
        return input_shape

    def _get_output_shape_onnxrt(self):
        output_details = self.interpreter.get_outputs()
        output_shape = {}
        for oup in output_details:
            output_shape.update({oup.name:oup.shape})
        #
        return output_shape
