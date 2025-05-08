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


import onnx
import onnx_graphsurgeon as gs


def change_batch_size(model):
    existing_batch_size = None
    new_batch_size = 1
    
    graph = gs.import_onnx(model)

    for input in graph.inputs:
        existing_batch_size = input.shape[0]
        input.shape[0] = new_batch_size

    for output in graph.outputs:
        output.shape[0] = new_batch_size

    for node in graph.nodes:
        for input in node.inputs:
            if input not in graph.inputs and input not in graph.outputs and isinstance(input, gs.Variable):
                input.shape = None
        for output in node.outputs:
            if output not in graph.outputs and output not in graph.outputs and isinstance(output, gs.Variable):
                output.shape = None

        if node.op == 'Reshape':
            shape = node.inputs[1]
            if shape.values[0] == existing_batch_size:
                shape.values[0] = new_batch_size

        if node.op == 'Concat':
            for input in node.inputs:
                if isinstance(input, gs.Constant) and input.values.shape[0] == existing_batch_size:
                    input.values = input.values[0:1]

    out_model = gs.export_onnx(graph)
    return out_model


def apply(transform, infile, outfile):
    model = onnx.load(infile)
    model = transform(model)
    onnx.save(model, outfile)
    onnx.shape_inference.infer_shapes_path(outfile, outfile)

model_name = "deit_small_patch16_224_simp.onnx"
out_model_name = model_name.replace(".onnx", "_bs1.onnx")

apply(change_batch_size, model_name, out_model_name)
onnx.shape_inference.infer_shapes_path(out_model_name, out_model_name)
