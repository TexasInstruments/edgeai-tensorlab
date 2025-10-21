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


import torch
import onnx_graphsurgeon as gs
from . import utils

def apply_func_to_all_tensors(func, tensors):
    if len(tensors) == 1:
        return func(tensors[0])
    return [func(tensor) for tensor in tensors]

onnx_2_torch = {
    'Max': torch.max,
    'Min': torch.min,
    'Mean': torch.mean,
}

def add_stat_op_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    assert node.op in onnx_2_torch, f'{node.name} with operator {node.op} is not implemented'
    func = onnx_2_torch[node.op]
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, torch.Tensor) for inp in node.inputs]
    if state.module_based:
        module = utils.WrappedModule(node.op, torch_module, apply_func_to_all_tensors )
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, (func, args))
    else:
        torch_nodes[node.name] = torch_graph.call_function(apply_func_to_all_tensors, (func, args), name=node.name)
    

