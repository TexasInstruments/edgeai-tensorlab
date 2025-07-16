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


def get_local_path(file_path, dest_dir):
    # sometimes, some http/https links can have '?' - remove the characters from there
    file_path = [f.split('?')[0] for f in file_path] if isinstance(file_path,(list,tuple)) else file_path.split('?')[0]
    if isinstance(file_path, (list,tuple)):
        file_path_local = [os.path.join(dest_dir, os.path.basename(m)) for m in file_path]
    else:
        file_path_local = os.path.join(dest_dir, os.path.basename(file_path))
    #
    return file_path_local


def file_exists(file_path):
    has_file = True
    if isinstance(file_path, (list,tuple)):
        for f in file_path:
            has_file = has_file and os.path.exists(f)
        #
    else:
        has_file = os.path.exists(file_path)
    #
    return has_file


def get_input_shape_onnx(onnx_model, num_inputs=1):
    input_shape = {}
    for input_idx in range(num_inputs):
        input_i = onnx_model.graph.input[input_idx]
        name = input_i.name
        shape = [dim.dim_value for dim in input_i.type.tensor_type.shape.dim]
        input_shape.update({name: shape})
    #
    return input_shape


def get_output_shape_onnx(onnx_model, num_outputs=1):
    output_shape = {}
    num_outputs = 1
    for output_idx in range(num_outputs):
        output_i = onnx_model.graph.output[output_idx]
        name = output_i.name
        shape = [dim.dim_value for dim in output_i.type.tensor_type.shape.dim]
        output_shape.update({name:shape})
    #
    return output_shape

