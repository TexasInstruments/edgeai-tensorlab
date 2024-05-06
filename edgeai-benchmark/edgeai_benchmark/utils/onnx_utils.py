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

# TI's adaptation of update_model_dims.
# Original code from onnx.tools.update_model_dims() has bug of iterating through all layers.
# But this version has less error checking code.

__all__ = ['onnx_update_model_dims']


def update_dim(tensor=None, new_dim_value=None, dim_idx=None):
    dim_proto = tensor.type.tensor_type.shape.dim[dim_idx]
    if isinstance(new_dim_value, str):
        dim_proto.dim_param = new_dim_value
    elif isinstance(new_dim_value, int):
        if new_dim_value >= 0:
            assert not (dim_proto.HasField('dim_value') and (dim_proto.dim_value != new_dim_value))
        else: #new_dim_value is negative. Not handled currently.
            assert False
    return


def onnx_update_model_dims(model, input_dims, output_dims):
    for input_name, input_dim_arr in input_dims.items():
        input_layer_tensor = [input_tensor for input_tensor in model.graph.input if input_tensor.name == input_name][0]
        for dim_idx, new_dim_value in enumerate(input_dim_arr):
            update_dim(tensor=input_layer_tensor, new_dim_value=new_dim_value, dim_idx=dim_idx)

    for output_name, output_dim_arr in output_dims.items():
        output_layer_tensor = [output_tensor for output_tensor in model.graph.output if output_tensor.name == output_name][0]
        for dim_idx, new_dim_value in enumerate(output_dim_arr):
            update_dim(tensor=output_layer_tensor, new_dim_value=new_dim_value, dim_idx=dim_idx)

    import onnx
    onnx.checker.check_model(model)
    return model
