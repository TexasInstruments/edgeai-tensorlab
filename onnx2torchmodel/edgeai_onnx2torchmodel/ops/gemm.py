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
from operator import getitem
from . import utils

def torch_gemm(a:torch.Tensor, b:torch.Tensor, c:torch.Tensor=None, alpha=1, beta=1, transA=False, transB=False):
    # return torch.mm
    if transA: 
        a = a.transpose(-2,-1)
    if transB:
        b = b.transpose(-2, -1)
    y = torch.matmul(a,b)
    if alpha != 1:
        y = alpha*y
    if c:
        if beta != 1:
            y = y + beta*c
        else:
            y = y + c
    return y

def add_gemm_2_torch_graph(state, node:gs.Node, torch_graph:torch.fx.Graph,  torch_nodes: dict[str,torch.fx.Node], torch_module:torch.nn.Module):
    types = [torch.nn.Parameter if inp.shape else torch.Tensor for inp in node.inputs]
    assert 2 <= len(node.inputs) <= 3, f'{node.name} with operator {node.op} should have 2 or 3 input, but got {len(node.inputs)}'
    args = [utils.get_input_from_node(inp, torch_graph,torch_nodes, torch_module, t) for inp,t in zip(node.inputs, types)]
    alpha = node.attrs.get('alpha', 1.0)
    beta = node.attrs.get('beta', 1.0)
    transA = node.attrs.get('transA', 0)==1
    transB = node.attrs.get('transB', 0)==1
    has_bias = len(node.inputs) == 3
    if not transA and alpha == 1 and beta == 1 and all(isinstance(t,c) for t,c in zip(node.inputs,(gs.Variable, gs.Constant, gs.Constant))):
        m,n = node.inputs[1].shape
        if transB:
            m,n = n,m
        module = torch.nn.Linear(m, n, bias=has_bias)
        weight = getattr(torch_module, node.inputs[1].name)
        if not transB:
            weight = torch.nn.Parameter(weight.T)
        module.weight = weight
        if has_bias:
            module.bias = getattr(torch_module, node.inputs[2].name)
        module.training = torch_module.training
        torch_module.add_module(node.name, module)
        torch_nodes[node.name] = torch_graph.call_module(node.name, tuple(args[0:1]),)
    else:
        torch_nodes[node.name] = torch_graph.call_function(torch_gemm, tuple(args), dict(alpha=alpha, beta=beta, transA=transA, transB=transB), name=node.name)
    