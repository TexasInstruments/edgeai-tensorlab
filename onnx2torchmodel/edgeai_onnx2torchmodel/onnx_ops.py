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
from .ops import utils, basic_ops_2_func_dict, custom_add_2_torch_graph

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
    
def get_torch_graph_module(graph:gs.Graph):
    inputs = list(graph.inputs)
    outputs = list(graph.outputs)
    
    op_2_func_dict = basic_ops_2_func_dict.copy()
    op_2_func_dict.update(custom_add_2_torch_graph)
    
    state = State()
    state.graph = graph
    torch_graph = torch.fx.Graph()
    root_module = torch.nn.Module()
    torch_nodes = {}
    for inp in inputs:
        torch_nodes[inp.name] = torch_graph.placeholder(inp.name)
    
    for node in graph.nodes:
        func = op_2_func_dict[node.op]
        func(state, node, torch_graph, torch_nodes, root_module)
    
    torch_outputs = []
    for out in outputs:
        torch_outputs.append(utils.get_input_from_node(out, torch_graph, torch_nodes, root_module))
    torch_nodes['outputs'] = torch_graph.output(tuple(torch_outputs))
    
    
    return torch.fx.GraphModule(root_module, torch_graph)