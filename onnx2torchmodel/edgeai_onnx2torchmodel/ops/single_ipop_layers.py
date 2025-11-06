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
import torch
from . import utils



onnx_to_torch = {
    'Abs': torch.abs,
    'Acos': torch.acos,
    'Acosh': torch.acosh,
    'Asin': torch.asin,
    'Asinh': torch.asinh,
    'Atan': torch.atan,
    'Atanh': torch.atanh,
    'Not': torch.logical_not,
    'BitwiseNot': torch.bitwise_not,
    'Ceil': torch.ceil,
    'Cos': torch.cos,
    'Cosh': torch.cosh,
    'Det' : torch.det,
    'Erf': torch.erf,
    'Log': torch.log,
    'Sin': torch.sin,
    'Sinh': torch.sinh,
    'Tan': torch.tan,
    'Tanh': torch.tanh,
    'Exp': torch.exp,
    'Floor': torch.floor,
    'IsNan': torch.isnan,
    'Neg': torch.neg,
    'Reciprocal': torch.reciprocal,
    'Round': torch.round,
    'Sign': torch.sign,
    'Sqrt': torch.sqrt
}

def add_node_2_torch_graph_1ip_1op(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    inp = node.inputs[0]
    if node.op in onnx_to_torch:
        args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, torch.nn.Parameter if inp.shape else torch.Tensor)]
        if state.module_based:
            module = utils.WrappedModule(node.name, node.op, torch_module, onnx_to_torch[node.op], args, )
            torch_module.add_module(node.name, module)
            args = [x for x in args if (isinstance(x, torch.fx.Node) and x.op != 'get_attr')]

            torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args))
        else:
            torch_nodes[node.name] = torch_graph.call_function(onnx_to_torch[node.op], tuple(args),  name=node.name)
    else:
        raise NotImplementedError