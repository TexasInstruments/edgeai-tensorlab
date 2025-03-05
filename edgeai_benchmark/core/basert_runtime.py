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


class BaseRuntimeWrapper:
    def __init__(self, **kwargs):
        if not hasattr(self, 'kwargs'):
            self.kwargs = kwargs
        else:
            self.kwargs.update(kwargs)

    def _get_input_details_onnx(self, interpreter, input_details=None):
        if input_details is None:
            properties = {'name':'name', 'shape':'shape', 'type':'type'}
            input_details = []
            model_input_details = interpreter.get_inputs()
            for inp_d in model_input_details:
                inp_dict = {}
                for p_key, p_val in properties.items():
                    inp_d_val = getattr(inp_d, p_key)
                    if p_key == 'type':
                        inp_d_val = str(inp_d_val)
                    #
                    if p_key == 'shape':
                        inp_d_val = list(inp_d_val)
                    #
                    inp_dict[p_val] = inp_d_val
                #
                input_details.append(inp_dict)
            #
        #
        return input_details

    def _get_output_details_onnx(self, interpreter, output_details=None):
        if output_details is None:
            properties = {'name':'name', 'shape':'shape', 'type':'type'}
            output_details = []
            model_output_details = interpreter.get_outputs()
            for oup_d in model_output_details:
                oup_dict = {}
                for p_key, p_val in properties.items():
                    oup_d_val = getattr(oup_d, p_key)
                    if p_key == 'type':
                        oup_d_val = str(oup_d_val)
                    #
                    if p_key == 'shape':
                        oup_d_val = list(oup_d_val)
                    #
                    oup_dict[p_val] = oup_d_val
                #
                output_details.append(oup_dict)
            #
        #
        return output_details

    def _get_input_details_tflite(self, interpreter, input_details=None):
        if input_details is None:
            properties = {'name':'name', 'shape':'shape', 'dtype':'type', 'index':'index'}
            input_details = []
            model_input_details = interpreter.get_input_details()
            for inp_d in model_input_details:
                inp_dict = {}
                for p_key, p_val in properties.items():
                    inp_d_val = inp_d[p_key]
                    if p_key == 'dtype':
                        inp_d_val = str(inp_d_val)
                    #
                    if p_key == 'shape':
                        inp_d_val = [int(val) for val in inp_d_val]
                    #
                    inp_dict[p_val] = inp_d_val
                #
                input_details.append(inp_dict)
            #
        #
        return input_details

    def _get_output_details_tflite(self, interpreter, output_details=None):
        if output_details is None:
            properties = {'name':'name', 'shape':'shape', 'dtype':'type', 'index':'index'}
            output_details = []
            model_output_details = interpreter.get_output_details()
            for oup_d in model_output_details:
                oup_dict = {}
                for p_key, p_val in properties.items():
                    oup_d_val = oup_d[p_key]
                    if p_key == 'dtype':
                        oup_d_val = str(oup_d_val)
                    #
                    if p_key == 'shape':
                        oup_d_val = [int(val) for val in oup_d_val]
                    #
                    oup_dict[p_val] = oup_d_val
                #
                output_details.append(oup_dict)
            #
        #
        return output_details
