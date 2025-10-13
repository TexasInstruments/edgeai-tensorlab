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


from typing import Any
import onnx_graphsurgeon as gs
import torch
import numpy as np
from .ops import utils, basic_ops_2_func_dict, custom_add_2_torch_graph

np_2_torch_type_mapping = utils.np_2_torch_type_mapping

class State:
    def __init__(self):
        self.data = {}
    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except:
            if name in self.data: return self.data[name]
    def __setattr__(self, name: str, value: Any) -> None:
        try:
            super().__setattr__(name, value)
        except:
            if hasattr(self, 'data'): 
                self.data[name] = value

def get_tensor_info(tensor:gs.Variable|gs.Constant):
    return dict(
        type = tensor.__class__.__name__,
        shape = tensor.shape,
        dtype = np_2_torch_type_mapping[tensor.dtype]
    )
    
def get_graph_info(graph:gs.Graph):
    inputs = list(graph.inputs)
    outputs = list(graph.outputs)
    nodes = list(graph.nodes)
    node_info = {}
    for node in nodes:
        node_info[node.name] = dict(
            op = node.op,
            attrs = node.attrs,
            inputs = {inp.name: get_tensor_info(inp) for inp in node.inputs},
            outputs = {out.name: get_tensor_info(out) for out in node.outputs}
        )
    input_info = {inp.name: get_tensor_info(inp) for inp in inputs} 
    output_info = {out.name: get_tensor_info(out) for out in outputs}
    return node_info, input_info, output_info

def check_convertable(graph:gs.Graph, op_2_func_dict=None, for_training=False)-> dict:
    op_2_func_dict = op_2_func_dict or basic_ops_2_func_dict
    error_dict = {}
    for node in graph.nodes:
        temp_graph = gs.Graph()
        temp_graph.nodes = [node]
        temp_graph.outputs = node.outputs
        temp_graph.inputs = [inp for inp in node.inputs if isinstance(inp, gs.Variable)]
        torch_nodes = {}
        temp_graph.opset = graph.opset
        torch_graph = torch.fx.Graph()
        state = State()
        state.training = for_training
        state.graph = temp_graph
        root_module = torch.nn.Module()
        root_module.training = for_training
        try:
            for inp in temp_graph.inputs:
                torch_nodes[inp.name] = torch_graph.placeholder(inp.name)
            
            for node in temp_graph.nodes:
                func = op_2_func_dict[node.op]
                func(state, node, torch_graph, torch_nodes, root_module)
            
            outputs = list(temp_graph.outputs)
            if len(outputs) == 1:
                output = utils.get_input_from_node(outputs[0], torch_graph, torch_nodes, root_module)
                torch_nodes['outputs'] = torch_graph.output(output)
            else:
                torch_outputs = [] 
                for out in outputs:
                    torch_outputs.append(utils.get_input_from_node(out, torch_graph, torch_nodes, root_module))
                torch_nodes['outputs'] = torch_graph.output(tuple(torch_outputs))
        except Exception as e:
            error_dict[(node.name, node.op)] = e
    return error_dict

def get_torch_graph_module(graph:gs.Graph, for_training=False):
    inputs = list(graph.inputs)
    
    op_2_func_dict = basic_ops_2_func_dict.copy()
    op_2_func_dict.update(custom_add_2_torch_graph)
    
    error_dict = check_convertable(graph, op_2_func_dict=op_2_func_dict,for_training=for_training)
    if error_dict:
        for name, op in error_dict:
            print(f'Failed to convert {name} with operator {op} because of error {error_dict[(name, op)]}')
        raise Exception('Failed to convert the model because of above errors')
    
    state = State()
    state.graph = graph
    state.training = for_training
    torch_graph = torch.fx.Graph()
    root_module = torch.nn.Module()
    root_module.training = for_training
    torch_nodes = {}
    for inp in inputs:
        torch_nodes[inp.name] = torch_graph.placeholder(inp.name)
    
    for node in graph.nodes:
        func = op_2_func_dict[node.op]
        func(state, node, torch_graph, torch_nodes, root_module)
    
    outputs = list(graph.outputs)
    if len(outputs) == 1:
        output = utils.get_input_from_node(outputs[0], torch_graph, torch_nodes, root_module)
        torch_nodes['outputs'] = torch_graph.output(output)
    else:
        torch_outputs = [] 
        for out in outputs:
            torch_outputs.append(utils.get_input_from_node(out, torch_graph, torch_nodes, root_module))
        torch_nodes['outputs'] = torch_graph.output(tuple(torch_outputs))
    
    
    torch_module = torch.fx.GraphModule(root_module, torch_graph)
    torch_module.node_info, torch_module.input_info, torch_module.output_info = get_graph_info(graph)
    return torch_module