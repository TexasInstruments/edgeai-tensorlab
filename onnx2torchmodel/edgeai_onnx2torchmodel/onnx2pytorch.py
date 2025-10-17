# Copyright (c) 2018-2025, Texas Instruments
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


import onnx_graphsurgeon as gs
import onnx
from . import onnx_ops


def simplify_graph(graph:gs.Graph):
    for idx, node in enumerate(graph.nodes):  
        if node.name.isnumeric() or not node.name:
            node.name = f'{node.op}_{idx}'
        else:
            node.name = node.name.replace('.','_').replace('/','_').replace(':','_')
    for name, tensor in graph.tensors().items():
        if name.isnumeric():
            tensor. name = f'tensor_{name}'
        else:
            tensor.name = tensor.name.replace('.','_').replace('/','_').replace(':','_')

    return graph


def remove_identity(graph:gs.Graph):
    nodes = [node for node in graph.nodes if node.op == 'Identity']
    for i, node in enumerate(nodes):
        if node.op == 'Identity':
            outputs = list(node.outputs)
            for out in outputs:
                for o in out.outputs:
                    if isinstance( node.inputs[0], gs.Constant ):
                        o.inputs[o.inputs.index(out)] = gs.Constant(out.name, node.inputs[0].values)
                    elif isinstance(node.inputs[0], gs.Variable):
                        o.inputs[o.inputs.index(out)] = gs.Variable(out.name, node.inputs[0].dtype, node.inputs[0].shape)
            graph.nodes.remove(node)
            if out in graph.outputs:
                graph.outputs.remove(out)


def convert(model_path, for_training=False):
    onnx_model = onnx.load(model_path)
    graph = gs.import_onnx(onnx_model)
    remove_identity(graph)
    simplify_graph(graph)
    try:
        model = gs.export_onnx(graph)
        onnx.save_model(model, model_path)
    except Exception as e:
        # print(f"Failed to convert model because of error {e}")
        pass

    torch_model = onnx_ops.get_torch_graph_module(graph, for_training=for_training)
    return torch_model
