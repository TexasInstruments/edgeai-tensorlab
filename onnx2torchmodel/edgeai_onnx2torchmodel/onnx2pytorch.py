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
import torch

import tempfile
import types
import atexit
import os
import sys
import importlib

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


pre_text = """
import torch

def create_forward_method():
"""

post_text = """
    return forward
"""

def change_forward_method(model:torch.fx.GraphModule):
    nodes = [node for node in model.graph.nodes if node.op !='placeholder']
    node_names = [node.name for node in nodes]
    code_lines = model.code.split('\n')[3:-1]
    new_forward = pre_text+'\n' + f'    {code_lines[0]}\n'
    node_count = 0
    for i, line in enumerate(code_lines[1:]):
        node = nodes[node_count]
        if ';' in line:
            end = line.index(';')
            line = line[:end]
        if '=' not in line:
            new_forward += f'    {line}\n'
            node_count += 1
            continue
        index = line.index(' =')
        if line[:index].lstrip() not in node_names:
            new_forward += f'    {line}\n'
            continue
        if node.op == 'call_function' and '(' in  line:
            i = line.index('= ')+2
            j = line.index('(')
            line = line[:i] + f'self.name_to_func_dict["{node.name}"]' + line[j:]
            new_forward += f'    {line}\n'
        else:
            new_forward += f'    {line}\n'
        node_count += 1
    
    return new_forward+'\n'+post_text

def import_file_folder(file_or_folder_name):
    if file_or_folder_name.endswith(os.sep):
        file_or_folder_name = file_or_folder_name[:-1]
    #
    parent_folder = os.path.dirname(file_or_folder_name)
    basename = os.path.splitext(os.path.basename(file_or_folder_name))[0]
    sys.path.insert(0, parent_folder)
    imported_module = importlib.import_module(basename, __name__)
    sys.path.pop(0)
    return imported_module


def new_del(self):
    if hasattr(self, 'forward_path'):
        os.remove(self.forward_path)
    super().__del__()

def convert(model_path, for_training=False, modify_forward=True):
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
    if modify_forward:
        torch_model.name_to_func_dict = {node.name:node.target for node in torch_model.graph.nodes if node.op == 'call_function'}
        
        new_forward_text = change_forward_method(torch_model)
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write(new_forward_text)
            f.flush()
            temp_py_path = f.name
        
        temp_module = import_file_folder(temp_py_path)
        new_forward_forward = temp_module.create_forward_method()
        torch_model.forward_path = temp_py_path
        torch_model.forward = types.MethodType(new_forward_forward, torch_model)
        torch_model.__del__ = types.MethodType(new_del, torch_model)
        atexit.register(lambda: os.remove(torch_model.forward_path) if hasattr(torch_model, 'forward_path') else None)
    # os.remove(temp_py_path)
    return torch_model
